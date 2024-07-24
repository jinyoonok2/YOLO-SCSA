### README

# Skin Cancer Detection Project

This repository contains code for training, evaluating, and running inference on a YOLOv8 model for skin cancer detection.

## Installation

### CUDA Installation

Ensure you have CUDA installed on your system. The CUDA version should match your system's settings. You can install CUDA with the following command:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```


For more information on CUDA installation and to find the appropriate version for your system, visit the [CUDA Local Installation Guide](https://pytorch.org/get-started/locally/).

### Install Dependencies

Clone the repository and navigate to the project folder. Then, install the required dependencies using `requirements.txt`:

```
pip install -r requirements.txt
```


Make sure to run the above command inside the project folder you have cloned.

## Usage

### Training

To train a YOLOv8 model, use the `train` task. You need to specify the model type, dataset path, image size, and model size. The experiment name will be generated automatically based on the model type and size.

Example command:
```
python model.py train --model_type scsa --data_path "./datasets/data_local.yaml" --img_size 512 --size s
```
This command trains the SCSA model with medium size on the dataset specified in `data_local.yaml` with an image size of 512.

Other Examples:
```
python model.py train --model_type basic --data_path "./datasets/data_local.yaml" --img_size 512 --size s
python model.py train --model_type rescbam --data_path "./datasets/data_local.yaml" --img_size 512 --size s
python model.py train --model_type sa --data_path "./datasets/data_local.yaml" --img_size 512 --size m
python model.py train --model_type ca --data_path "./datasets/data_local.yaml" --img_size 512 --size m
```
The weights and the results will be saved under logs folder of the current project directory.

You can find and download our customized HAM10000 dataset and the types of the models that were trained on that dataset from [here](https://drive.google.com/drive/folders/17glL50zG9XtoJaM6t0cUgPZhdufNYv81?usp=drive_link).

### Evaluation

To evaluate a specific trained model, use the `evaluate` task. You need to provide the path to the model and the dataset path.

Example command:

```
python model.py evaluate --model_path "./logs/SCSA(m)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
```

This command evaluates the SCSA(M) model on the dataset specified in `data_local.yaml` with an image size of 512.

Other Examples:
```
python model.py evaluate --model_path "./logs/Basic(s)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
python model.py evaluate --model_path "./logs/ResCBAM(s)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
python model.py evaluate --model_path "./logs/SA(s)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
python model.py evaluate --model_path "./logs/CA(m)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
```

### Inference

To run inference using one or more trained models, use the `inference` task. You need to provide the paths to the model files, the path to the images, and the path to the labels.

Example command:

```
python model.py inference --model_paths "./logs/Basic(s)/weights/best.pt" "./logs/SCSA(m)/weights/best.pt" "./logs/ResBlockCBAM(m)/weights/best.pt" --images_path "./datasets/valid/images" --labels_path "./datasets/valid/labels" --img_size 512
```

This command runs inference using the specified models on the validation images and labels with an image size of 512.

Since we used up to 5 models in our experiment, you can use up to 5 models for the inference for the unique colors of bounding box and compare their prediction results.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Make sure to follow these instructions carefully to set up and run the project successfully. If you encounter any issues, please refer to the documentation or raise an issue on the repository.
