## Data storage
The data storage location should follow the following method
- Datasets/
  - EADC/
    - sub1/
      - sub1_img.nii.gz
      - sub1_mask.nii.gz
    - ...
    - sub135/
        - sub135_img.nii.gz
        - sub135_mask.nii.gz 
    - train.txt
    - valid.txt
    - test.txt

Note:The label is not matched the image in th fllowing subjects:002_S_0938 (sub8), 007_S_1304 (sub35), 016_S_4121 (sub65), 029_S_4279 (sub85), 136_S_0429 (sub134).
