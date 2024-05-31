# Image Classifier using PyTorch

This repository contains code to train and predict image classifications using a neural network built with PyTorch. The project consists of two main scripts: `train.py` for training the model and `predict.py` for making predictions with the trained model.

## Table of Contents
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Predicting Image Classes](#predicting-image-classes)
- [File Descriptions](#file-descriptions)
- [Example Usage](#example-usage)
- [License](#license)

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ImageClassifier.git
cd ImageClassifier
pip install -r requirements.txt
```

Ensure you have PyTorch installed. You can install it using:
```bash
pip install torch torchvision
```

## Training the Model
Use the train.py script to train your image classifier. You can specify various hyperparameters and options:
```bash
python train.py data_directory --save_dir save_directory --arch model_architecture --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu
```

## Arguments
data_directory: Path to the dataset directory.
--save_dir: Directory to save the checkpoint.
--arch: Model architecture from torchvision.models (e.g., densenet121, vgg13).
--learning_rate: Learning rate for training.
--hidden_units: Number of hidden units in the classifier.
--epochs: Number of epochs to train.
--gpu: Use GPU for training if available.

## Predicting Image Classes
Use the predict.py script to predict the class of an input image using a trained model checkpoint:

```bash
python predict.py path_to_image checkpoint --top_k 5 --category_names cat_to_name.json --gpu
```
## Arguments
path_to_image: Path to the image file.
checkpoint: Path to the model checkpoint file.
--top_k: Return top K most likely classes.
--category_names: Path to JSON file mapping categories to real names.
--gpu: Use GPU for inference if available.
File Descriptions
train.py: Script to train the image classifier model.
predict.py: Script to predict image classes using a trained model.
cat_to_name.json: JSON file mapping category labels to real names.
requirements.txt: List of required packages and dependencies.

## Example Usage
### Training
```bash
python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu
```

### Predicting
python predict.py flowers/test/1/image_06752.jpg checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu

### Output
The predict.py script will output a DataFrame with the predicted classes and their respective probabilities.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
