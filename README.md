### README

# Skin Cancer Detection Project

This repository contains code for training, evaluating, and running inference on a YOLOv8 model for skin cancer detection.

## Installation

### CUDA Installation

Ensure you have CUDA installed on your system. The CUDA version should match your system's settings. You can install CUDA with the following command:

'''
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
'''


For more information on CUDA installation and to find the appropriate version for your system, visit the [CUDA Installation Guide](https://pytorch.org/get-started/locally/).

### Install Dependencies

Clone the repository and navigate to the project folder. Then, install the required dependencies using `requirements.txt`:

'''
pip install -r requirements.txt
'''


Make sure to run the above command inside the project folder you have cloned.

## Usage

### Training

To train a YOLOv8 model, use the `train` task. You need to specify the model type, dataset path, image size, and model size. The experiment name will be generated automatically based on the model type and size.

Example command:


'''
python model.py train --model_type scsa --data_path "./datasets/data_local.yaml" --img_size 512 --size m
'''


This command trains the SCSA model with medium size on the dataset specified in `data_local.yaml` with an image size of 512.

### Evaluation

To evaluate a specific trained model, use the `evaluate` task. You need to provide the path to the model and the dataset path.

Example command:


'''
python model.py evaluate --model_path "./logs/SCSA(M)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
'''

This command evaluates the SCSA(M) model on the dataset specified in `data_local.yaml` with an image size of 512.

### Inference

To run inference using one or more trained models, use the `inference` task. You need to provide the paths to the model files, the path to the images, and the path to the labels.

Example command:

'''
python model.py inference --model_paths "./logs/Basic(s)/weights/best.pt" "./logs/SCSA(m)/weights/best.pt" "./logs/ResBlockCBAM(m)/weights/best.pt" --images_path "./datasets/valid/images" --labels_path "./datasets/valid/labels" --img_size 512
'''


This command runs inference using the specified models on the validation images and labels with an image size of 512.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Make sure to follow these instructions carefully to set up and run the project successfully. If you encounter any issues, please refer to the documentation or raise an issue on the repository.
