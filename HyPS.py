import math
import torch
from torch import nn
import torch.nn.functional as F

def recursive_getattr(model, module_name):
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output

def recursive_setattr(model, module_name, module):
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)

class LinearLayer_Adapter(nn.Module):
    def __init__(self, R_W, D_W, Adapter_dim=0, Adapter_scaling=1, Adapter_droppout=0, bias=None):
        super(LinearLayer_Adapter, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.R_W = R_W.detach().to(self.device)
        self.D_W = nn.Parameter(D_W.to(self.device)) 
        self.bias = bias.to(self.device) if bias is not None else None

        if Adapter_dim <= 0:
            raise ValueError("Adapter dimension should be larger than 0")

        rows, columns = D_W.shape
        self.Adapter_right_weight = nn.Parameter(torch.zeros(rows, Adapter_dim).to(self.device))
        self.Adapter_left_weight = nn.Parameter(torch.zeros(Adapter_dim, rows).to(self.device))
        self.Adapter_scaling = Adapter_scaling / Adapter_dim

        if Adapter_droppout > 0:
            self.Adapter_dropout = nn.Dropout(Adapter_droppout)
        else:
            self.Adapter_dropout = nn.Identity()

        self.reset_parameters()

        self.R_W.requires_grad = False

        self.fuse_Adapter = False

    def eval(self):
        self.Adapter_dropout.eval()

    def train(self, mode=True):
        self.Adapter_dropout.train(mode)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.Adapter_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.Adapter_left_weight)

    def forward(self, input):
        input = input.to(self.device) 

        if self.fuse_Adapter:
            return F.linear(input, self.R_W + self.D_W, self.bias)
        else:
            self.Adapter_right_weight = self.Adapter_right_weight.to(self.device)
            self.Adapter_left_weight = self.Adapter_left_weight.to(self.device)

            activated_weight = torch.relu(self.Adapter_left_weight)
            output = F.linear(input, self.R_W + self.D_W, self.bias)
            Ada_output = (self.Adapter_dropout(output) @ self.Adapter_right_weight @ activated_weight) * self.Adapter_scaling
            return output + Ada_output

def convert_linear_layer_to_Adapter_with_SVD(model, part_module_name, r, Adapter_dim=0, Adapter_scaling=1, Adapter_droppout=0):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        U, S, V = torch.svd(module.weight)
        
        U1 = U[:, r:]
        S1 = torch.diag(S[r:])
        V1 = V[:, r:]

        U2 = U[:, :r]
        S2 = torch.diag(S[:r])
        V2 = V[:, :r]

        R_W = U1 @ S1 @ V1.T
        D_W = U2 @ S2 @ V2.T

        tmp = LinearLayer_Adapter(
            R_W, D_W, Adapter_dim, Adapter_scaling, Adapter_droppout,
            module.bias).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model

def only_optimize_Adapter_and_D_W_parameters(model, force_optimize_params=[]):
    for name, param in model.named_parameters():
        if "Adapter_right_weight" in name or "Adapter_left_weight" in name or name in force_optimize_params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def make_model_gradient_checkpointing_compatible(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model

def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        Adapter_lr=5e-4,
        no_decay_name_list=[
            "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
            "ln_f.weight"
        ],
        Adapter_name_list=["Adapter_right_weight", "Adapter_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in Adapter_name_list))
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in Adapter_name_list))
            ],
            "weight_decay": weight_decay,
            "lr": Adapter_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups



