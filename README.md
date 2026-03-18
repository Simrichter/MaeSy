# MaeSy

Leveraging Pretraining and Synthetic Finetuning to Train Data-Efficient Vision Transformers in the Context of Robot Soccer

## Overview

[//]: # (MaeSy is a Python framework for training custom Vision Transformers for object detection. The framework provides a modular architecture with separate components for dataset management, model architecture, training, and evaluation.)

[//]: # (## Features)

[//]: # ()
[//]: # (- 🎯 **Custom Vision Transformer Architecture**: Built from scratch with attention mechanisms optimized for object detection)

[//]: # (- 🔥 **Pretraining Support**: Two pretraining methods - Masked Autoencoder &#40;MAE&#41; and supervised classification pretraining)

[//]: # (- 📦 **Dataset Management**: Easy-to-use tools for downloading, managing, and preprocessing datasets in COCO format)

[//]: # (- 🚀 **Training Pipeline**: Complete training infrastructure with mixed precision support, learning rate scheduling, and checkpointing)

[//]: # (- 📊 **Evaluation Tools**: Comprehensive evaluation metrics including mAP, precision, recall, and visualization utilities)

[//]: # (- 🔧 **Modular Design**: Clean separation of concerns with independent modules for each major component)

## Installation

### From Source

```bash
git clone https://github.com/Simrichter/MaeSy.git
cd MaeSy
```
In a virtual environment, install the package (in editable mode):
```bash
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

[//]: # (_Main dependencies:)

[//]: # (- PyTorch >= 2.0.0)

[//]: # (- torchvision >= 0.15.0)

[//]: # (- numpy >= 1.21.0)

[//]: # (- Pillow >= 9.0.0)

[//]: # (- pycocotools >= 2.0.4_)

## Framework Architecture

MaeSy consists of five main modules:

### 1. Dataset Module (`maesy dataset`)

Handles dataset downloading, rosbag log extraction, clustering and data loading

Usage:
```bash
maesy dataset -h
```

### 2. Train Module (`maesy train`)

Runs varying training pipelines specified in maesy/training/train_setups
Includes classification, MAE pretraining and object detection training.
Starting from pretrained backbones is supported as well as resuming from training

Usage:
```bash
maesy train -h
```

### 3. Evaluate Module (`maesy evaluate`)
Provides utilities for running model inference, or visualizing datasets and predictions

Usage:
```bash
maesy evaluate -h
```

## Quick Start

### Training a Model with wandb logging

```bash
maesy train od --dataset path/to/dataset --wandb
```


## Model Architecture

All models inherit from the base_model class that requires a base_backbone and a base_head.

## Dataset Format

MaeSy supports YOLO format datasets and expects the dataset root as parameter.
Your dataset should have:

```
dataset_root/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│   ├── image1.txt
│   └── image2.txt
└── val/
    ├── images/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── labels/
        ├── image1.txt
        └── image2.txt
```
The individual label files should be in the format:
```
class_id x_center y_center width height
class_id x_center y_center width height
...
```
All values are normalized to the range [0, 1].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MaeSy in your research, please cite:

```bibtex
@software{maesy2026,
  title={MaeSy: Vision Transformer Framework for Object Detection},
  author={MaeSy Team},
  year={2026},
  url={https://github.com/Simrichter/MaeSy}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This framework is designed for data-efficient Vision Transformer training in the context of robot soccer
