import math
import torch
from torch import nn
import torch.nn.functional as F

def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output

def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)

class LinearLayer_cps(nn.Module):
    # an simple implementation of Adapter
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 cps_dim=0,
                 cps_scaling=1,
                 cps_droppout=0,
                 bias=None):
        super(LinearLayer_cps, self).__init__()
        self.weight = weight
        self.bias = bias

        if cps_dim <= 0:
            raise ValueError(
                "You are training to use Adapter, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.cps_right_weight = nn.Parameter(torch.zeros(
            rows,
            cps_dim))  # apply transpose so in forward we do not need to
        self.cps_left_weight = nn.Parameter(torch.zeros(cps_dim, rows))

        self.cps_right_weight1 = nn.Parameter(torch.zeros(columns,cps_dim))
        self.cps_left_weight1 = nn.Parameter(torch.zeros(cps_dim,rows))


        self.cps_scaling = cps_scaling / cps_dim

        if cps_droppout > 0:
            self.cps_dropout = nn.Dropout(cps_droppout)
        else:
            self.cps_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        self.fuse_cps = False

    def eval(self):
        self.cps_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.cps_dropout.train(mode)


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.cps_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.cps_left_weight)
        nn.init.kaiming_uniform_(self.cps_right_weight1, a=math.sqrt(5))
        nn.init.zeros_(self.cps_left_weight1)



    def forward(self, input):
        if self.fuse_cps:
            return F.linear(input, self.weight, self.bias)
        else:
            activate_weight = torch.relu(self.cps_left_weight)
            output = F.linear(input, self.weight,self.bias)
            Ada_output = (self.cps_dropout(output) @ self.cps_right_weight @ activate_weight) * self.cps_scaling
            lora_output = (self.cps_dropout(input) @ self.cps_right_weight1
                              @ self.cps_left_weight1) * self.cps_scaling
            return output + Ada_output + lora_output

# convert the linear layer to Adapter
def convert_linear_layer_to_cps(model,
                                 part_module_name,
                                 cps_dim=0,
                                 cps_scaling=1,
                                 cps_droppout=0):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_cps(
            module.weight, cps_dim, cps_scaling, cps_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model



def only_optimize_cps_parameters(model, force_optimize_params=[]):
    # turn off the gradient of all the parameters except the Adapter parameters
    for name, param in model.named_parameters():
        if "cps_right_weight" in name or "cps_left_weight" in name or "cps_right_weight1" in name or "cps_left_weight1" in name or name in force_optimize_params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for Adapter-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model

def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        cps_lr=5e-4,
        no_decay_name_list=[
            "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
            "ln_f.weight"
        ],
        cps_name_list=["cps_right_weight", "cps_left_weight","cps_left_weight1","cps_right_weight1"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in cps_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in cps_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                cps_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups
        
    
    
