## DSR - A dual subspace re-projection network for surface anomaly detection

This repository contains the implementation of **DSR** as proposed in the paper **_DSR -- A dual subspace re-projection network for surface anomaly detection_**

### Requirements
You can create a conda environment with: 
**conda create --name <env> --file requirements.txt**

But the main packages used are:

PyTorch 1.11, opencv-python, sklearn, numpy


### DSR evaluation on MVTec
#### Downloading the MVTec benchmark
Set up the MVTec AD benchmark by downloading it from: <https://www.mvtec.com/company/research/datasets/mvtec-ad>,
and extracting it. For training and evaluation pass the base directory of the extracted files (for example ./data/mvtec/) 
as the **--data_path** argument.

#### Download the pretrained models
Download the pretrained models and extract the zip so that the checkpoints folder will be located in the base directory of this repository.

Download link: <https://drive.google.com/file/d/15plhikrUjYCcx23JVxxBKb-HBwKAb8UK/view?usp=sharing>


#### Running the evaluation
```shell
#BASE_PATH -- the base directory of mvtec
#i -- the gpu id used for evaluation
python test_dsr.py $i $BASE_PATH DSR
```

### Training DSR
```shell
#BASE_PATH -- the base directory of mvtec
#OUT_PATH -- where the trained models will be saved
#i -- the index of the object class in the obj_batch list in train_dsr.py
python train_dsr.py --gpu_id 0 --obj_id $i --lr 0.0002 --bs 8 --epochs 100 --data_path $BASE_PATH --out_path $OUT_PATH
```
