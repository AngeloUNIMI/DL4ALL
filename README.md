# DL4ALL

Python/PyTorch source code for the paper:

	A. Genovese, V. Piuri, K. N. Plataniotis, and F. Scotti, 
    "DL4ALL: Multi-task cross-dataset transfer learning for Acute Lymphoblastic Leukemia detection", 
    IEEE Access, 2023.
	
Project page:

[https://iebil.di.unimi.it/cnnALL/index.htm](https://iebil.di.unimi.it/cnnALL/index.htm)
    
Outline:
![Outline](https://iebil.di.unimi.it/cnnALL/imgs/outline_dl4all.jpg "Outline")

Citation:

	@Article{access23,
    author = {A. Genovese and V. Piuri and K. N. Plataniotis and F. Scotti},
    title = {DL4ALL: Multi-task cross-dataset transfer learning for Acute Lymphoblastic Leukemia detection},
    journal = {IEEE Access},
    pages = {1-17},
    year = {2023},
    note = {Accepted},}

Main files:

- 1a_PyTorch_CNN_Regular: DL4ALL_ResNet18_Regular
- 1b_PyTorch_CNN_Self: DL4ALL_ResNet18_Self
- 1c_PyTorch_CNN_Greedy: DL4ALL_ResNet18_Greedy
- 2a_PyTorch_ViT_Regular: DL4ALL_ViT_Regular
- 2b_PyTorch_ViT_Self: DL4ALL_ViT_Self
- 2c_PyTorch_ViT_Greedy: DL4ALL_ViT_Greedy

Prerequisites:
https://github.com/lukemelas/PyTorch-Pretrained-ViT
pip install pytorch_pretrained_vit
    
Required files:
    
    - ADP/img_res_1um_bicubic/ <br/>
    ADP database, split in patches, obtained following the instructions at: 
    https://www.dsp.utoronto.ca/projects/ADP/ 
    
    - ADP/ADP_EncodedLabels_Release1_Flat.csv
    file containing the labels of the ADP database, obtained following the instructions at:
    https://www.dsp.utoronto.ca/projects/ADP/ 
    
    - C_NMC 2019 database
    https://www.kaggle.com/datasets/avk256/cnmc-leukemia
    https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223
        
The databases used in the paper can be obtained at:

- Atlas of Digital Pathology (ADP)<br/>
https://www.dsp.utoronto.ca/projects/ADP/

    Mahdi S. Hosseini, Lyndon Chan, Gabriel Tse, Michael Tang, Jun Deng, Sajad Norouzi, Corwyn Rowsell, Konstantinos N. Plataniotis, Savvas Damaskinos
    "Atlas of Digital Pathology: A Generalized Hierarchical Histological Tissue Type-Annotated Database for Deep Learning"
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11747-11756

- C-NMC_Leukemia <br/>
https://www.kaggle.com/datasets/avk256/cnmc-leukemia
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223

    Rahul Duggal, Anubha Gupta, Ritu Gupta, and Pramit Mallick, "SD-Layer: Stain Deconvolutional Layer for CNNs in Medical Microscopic Imaging," In: Descoteaux M., Maier-Hein L., Franz A., Jannin P., Collins D., Duchesne S. (eds) Medical Image Computing and Computer-Assisted Intervention − MICCAI 2017, MICCAI 2017. Lecture Notes in Computer Science, Part III, LNCS 10435, pp. 435–443. Springer, Cham. DOI: https://doi.org/10.1007/978-3-319-66179-7_50.
    
