# Nuclei-segmentation

This repository gathers the code for nuclei-segmentation from the [in-class CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5).

We use [Detectron2](https://github.com/facebookresearch/detectron2), a Python API provided by Facebook research for Mask R-CNN, based on the PyTorch framework, to train our model.
In this competition, we use  Mask R-CNN on ResNet-101 and ResNeXt-101(32x8d) backbone these two models and analysis the results.

## Reproducing Submission
We need to do some pre-preparation for training and testing on our custom dataset.

To reproduce my submission without retrainig, do the following steps:
1. [Requirement](#Requirement)
2. [Repository Structure](#Repository-Structure)
3. [Dataset setting](#Dataset-setting)
4. [Inference](#Inference)

## Hardware

Ubuntu 20.04.3 LTS

Intel® Core™ i7-9700KF CPU @ 3.60GHz × 8

GeForce GTX 3090 32G


## Requirement
All requirements should be:

```env
$ virtualenv detectron2 --python=3.8
$ source ./detectron2/bin/activate
$ cd Nuclei-segmentation
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
$ pip install opencv-python
```

Official images can be downloaded from [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5#participate-get_data)


## Repository Structure

The repository structure is:
```
Nuclei-segmentation(root)
  +-- configs                    # all configs(hyperparameters) used in the program 
  |   +-- COCO-InstanceSegmentation
      |   +-- mask_rcnn_R_101_FPN_3x.yaml
      |   +-- mask_rcnn_X_101_32x8d_FPN_3x.yaml
  |   +-- Base-RCNN-FPN.yaml
  +-- dataset                    # all training and testing files
  |   +-- coco
      |   +-- annotations        # json files
          |   +-- test_img_ids.json   
          |   +-- test.json   
          |   +-- train.json   
      |   +-- test               # testing set .png  
          |   +-- TCGA-50-5931-01Z-00-DX1.png  
          |   +-- TCGA-A7-A13E-01Z-00-DX1.png 
          |   +-- ......
      |   +-- train              # origin training data (each image have one folder)
      |   +-- trainval           # training set .png
          |   +-- TCGA-18-5592-01Z-00-DX1.png   
          |   +-- TCGA-21-5784-01Z-00-DX1.png 
          |   +-- ...... 
      |   +-- parse_test.py      # parse testing json file
      |   +-- parse_train.py     # parse training json file
  +-- input_img                  # save training images with segmentation there
  +-- output_img                 # save testing images with segmentation there
  +-- output                     # save training model and matrics.
  +-- visual_test.py             # output testing images with segmentation
  +-- visual_train.py            # output training images with segmentation
  +-- inference.py               # model prediction and reproduce my submission file
  +-- train.py                   # for training model
```

## Dataset setting

You can use ```parse_train.py``` to parse training images to .json file and also use ```parse_test.py```  to parse testing images to .json file.

```py
$ python dataset/coco/parse_test.py
$ python dataset/coco/parse_train.py
```

## Training

To train the model, run this command:

```py
$ python train.py
```

Trained model will be saved as ```output/R101/{1, 2, 3,...}/model_final.pth```

All mAP of experiments will be written in [Results](#Results).

## Inference

Please download [this model]() if you want to reproduce my submission file, and run above codes.

To reproduce my submission file or test the model you trained, run:

```py
$ python inference.py
```

Prediction file will be saved as ```root/answer.json```

## Results

Our model achieves the following performance on :

ResNet-101 Result:

| No.         | 1                     | 2                     | 3                     | 4                     | 5                     | 6                     | 7                     | 8                     | 9                     | 10                 | 11                   |  
|:-----------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:------------------:|:--------------------:|
| model       | R101                  | R101                  | R101                  | R101                  | R101                  | R101                  | R101                  | R101                  | R101                  | R101               | R101                 |   
| mAP         | 0.223274 | 0.219256 | 0.208042 | 0.221926 | 0.226062 | 0.211141 | 0.242515 | **0.24355** | 0.232426 | 0.233071 | 0.242185 |   
| Batch size  | 128 | **512** | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 |   
| Anchor size | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | **8, 16, 32, 64, 128** | **16, 32, 64, 128, 256** |   
| Iteration   | 300000 | 300000 | 300000 | **500000** | **50000** | **200000** | **6000** | **5000** | **4000** | 5000 | 5000 |   
| Image size  | 800 | 800 | **1200** | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 |   
| Note(change)        |                      | batch size            | image size            | iteration             | iteration             | iteration             | iteration             | iteration             | iteration             | anchor size        | anchor size          |   

ResNeXt-101(32x8d) Result:

| No.         | 12                    | 13                    | 14                    | 15                    |   
|:-----------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| model       | **X101**                  | X101                 | X101                  | X101                  |   
| mAP         | 0.228375 | 0.225095 | 0.243045 | 0.231643 |   
| Batch size  | 128 | 128 | 128 | 128 |   
| Anchor size | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 | 32, 64, 128, 256, 512 |   
| Iteration   | **50000** | **100000** | **5000** | **3000** |   
| Image size  | **1000** | 1000 | 1000 | 1000 |   
| Note        | iteration             | iteration             | iteration             | iteration             |   


## Reference
[1] [Detectron2](https://github.com/facebookresearch/detectron2)

[2] [pycocotools](https://github.com/cocodataset/cocoapi/issues/131)

[3] [RLE encode](https://github.com/facebookresearch/detectron2/issues/347)
