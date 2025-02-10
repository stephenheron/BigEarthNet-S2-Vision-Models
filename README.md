# BigEarthNet S2 Vision Models

An experimental repo for classifiying  [BigEarthNet with Sentinel-2 Image Patches](https://bigearth.net/). 

Used as a learning resource for building a selection of image models.

I wanted to train locally so all models were trained on a Nvidia 4070 Super.

## ViT
ViT model from scratch based on this [guide](https://comsci.blog/posts/vit).

Total Parameters: 6,401,811

| Metric | Value |
|--------|-------|
| Precision | 0.7254 |
| Recall | 0.7819 |
| F1 Score | 0.7526 |
| Exact Match Ratio | 0.2725 |
| Images per second (Test) | 81,546 |
| Training time | 8.52 hrs |

## CNN
A fine tuned CNN based model using [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html). 

Total Parameters: 10,506,195

| Metric | Value |
|--------|-------|
| Precision | 0.7946 |
| Recall | 0.6919 |
| F1 Score | 0.7397 |
| Exact Match Ratio | 0.2881 |
| Images per second (Test) | 57,467 |
| Training time | 20.46 hrs |

## ConvNeXt
A fine tuned ConvNeXt model using [ConvNeXt](https://pytorch.org/vision/main/models/convnext.html)

Total Parameters: 46,763,827

| Metric | Value |
|--------|-------|
| Precision | 0.7451 |
| Recall | 0.7906 |
| F1 Score | 0.7672 |
| Exact Match Ratio | 0.2949 |
| Images per second (Test) | 25,040 |
| Training time | 6 hrs |
