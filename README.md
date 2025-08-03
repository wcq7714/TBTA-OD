## Environment

* python==3.8
* torch==1.11.0
* torchvision==0.12.0
* mmcv-full==1.5.1
* diffusers==0.30.3
* [mmdetection v2.24.1](https://github.com/open-mmlab/mmdetection/tree/v2.24.1)
* [mmsegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0)

## Stable Diffusion Model

```
huggingface-cli download --resume-download CompVis/stable-diffusion-v1-4 --local-dir .
```

## Datasets

Object detection: get VOC and COCO datasets under `/data` folder.
```
cd data
bash get_voc.sh
bash get_coco.sh
```

## Object detection attack

1. Download and place `mmdetection` folder under TBTA-OD directory.

2. run ```python mmdet_model_info.py``` to download pre-trained models from MMCV.

3. Download 'CompVis/stable-diffusion-v1-4' model.

4. run ```python tbtaod.py``` to perform PGD based attacks on object detection.

5. run ```python tbtaod-mifgsm.py``` to perform MI-FGSM based attacks on object detection.
