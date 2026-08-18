# MaeSy

#### Leveraging Pretraining and Synthetic Finetuning to Train Data-Efficient Vision Transformers in the Context of Robot Soccer

This framework is designed for data-efficient Vision Transformer training in the context of robot soccer

## Installation
##### If you intend to train on GPUs, you have to install cuda drivers and matching torch wheel separately

Download source from git (or unpack zip archive)
```bash
git clone https://github.com/Simrichter/MaeSy.git
```
[Optional] source a virtual environment

Install the package
(use the editable flag '-e' if you intend to modify code):
```bash
pip install MaeSy
```
## Usage
The maesy framework is organized hierarchically.
Use the -h flag at every level to obtain the parameter options of a (sub)module

The modules are structured as follows:
```
maesy
├── train
│   ├── od: Object Detection
│   ├── mae: Masked Autoencoder Training of a backbone
│   ├── cl: Classification (experimental)
│   └── pc: Patch Classification (experimental)
├── evaluate
|   ├── test: Evaluate model on a dataset
|   ├── compare: compare between multiple test results
|   ├── infer: Run inference on images
|   └── visualize: Visualize a dataset or predictions from inference
├── dataset
|   ├── extract_log: Extract images from ROSbag log file
|   ├── extract_patches: Extract object patches from a dataset
|   ├── download_data: Download and uncompress datasets from a url
|   ├── create: Create a ready-to-use MaeSyDataset from (labeled) image data
|   └── convert: Conversion between dataset formats
├── export: exports a model to onnx format
└── bulk_execute: Automatically execute maesy-commands from a text file
```

Example commands:
```
maesy train od rt-detr6 --dataset data/Cvat --backbone mae_checkpoints/bright-shadow-11/best_model.pth --enable-denoising --enable-line-detection --wandb --name "rtdetr6(i+mae+cvat)"

maesy train od od_checkpoints/rtdetr6(i+mae+wb,u)/best_model.pth --dataset data/Cvat --enable-denoising --enable-line-detection --wandb --name "rtdetr6(i+mae+wb,u+cvat)" --finetune

maesy export od_checkpoints/rtdetr6(i+mae+wb,u)/best_model.pth --name exported_model --enable-line-detection --line-class-id 3
```

## MaeSyDataset Format

MaeSy uses the MaeSyDataset format (extended from YOLO format) and expects the path to the dataset root or its 'dataset.yaml' as parameter when specifying datasets.
By default, datasets are created in a 'data/' subdirectory of the working directory.

MaeSyDatasets have the following structure:

```
dataset_root/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
└── val/
|   ├── images/
|   │   ├── image1.jpg
|   │   └── image2.jpg
|   └── labels/
|       ├── image1.txt
|       └── image2.txt
└── dataset.yaml
```

The individual label files should be in the format:
```
class_id param1 param2 param3 ...
class_id param1 param2 param3 ...
...
```
All values are normalized to the range [0, 1].

The dataset.yaml (can be renamed) should have the following contents:

```
box_format: Format of bounding box representations (xyxy or cxcywh)
lines: Name of line class
names:
- Names
- of
- classes
nc: Total number of classes
path: path/to/dataset/root
test: test (only change these if a different structure is used)
train: train
val: val
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MaeSy in your research, please cite:

```bibtex
@software{maesy2026,
  title={Leveraging Pretraining and Synthetic Finetuning to Train Data-Efficient Vision Transformers in the Context of Robot Soccer},
  author={Simon Richter},
  year={2026},
  url={https://github.com/Simrichter/MaeSy}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
