# [ECCV22] Bi-directional Contrastive Learning for Domain Adaptive Semantic Segmentation

This is a PyTorch implementation of [Bi-directional Contrastive Learning for Domain adaptive Semantic Segmentation].  

<figure>
    <img src="./overview.png" alt="DASS">
</figure>

### Requirements

To install requirements:

- Python 3.6
- Pytorch 1.4.0

## Getting Started

1. Download the dataset. 
2. Download the ImageNet-pretrained Model [[Link](https://drive.google.com/open?id=13kjtX481LdtgJcpqD3oROabZyhGLSBm2)].

## Training

Train the source-only model:

```train
python so_run.py
```

Train our model:

```train
python run.py
```

## Evaluation

To perform evaluation on single model:

```eval
python eval.py --frm model.pth --single
