# Fashion MNIST Classification with Neural Network

## Introduction

This project implements a neural network to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset consists of 28x28 grayscale images of 10 fashion categories, such as T-shirts, trousers, and dresses. The goal is to train a neural network to accurately classify these images into their respective categories.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

## Project Overview

This repository contains the code and resources for training and evaluating a neural network model on the Fashion MNIST dataset. The project includes data preprocessing, model architecture, training and evaluation scripts, and visualizations of the results.

## Dataset

The Fashion MNIST dataset is a collection of 70,000 images, split into 60,000 training samples and 10,000 test samples. Each image is a 28x28 grayscale picture associated with a label from 10 classes:

| Label | Description   |
|-------|---------------|
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

For more information, visit the [Fashion MNIST GitHub repository](https://github.com/zalandoresearch/fashion-mnist).

## Model Architecture

The neural network implemented in this project consists of the following layers:

1. Input Layer: 28x28 neurons, one for each pixel in the image.
2. Hidden Layers:
3. Output Layer: Dense layer with 10 neurons, softmax activation.

The model uses categorical cross-entropy as the loss function and the Adam optimizer for training.


## Installation

To run the code, ensure you have Python 3.7 or higher installed, along with the necessary libraries:

```Python
```

requirements.txt should include:

```
tensorflow
numpy
matplotlib
```

## Usage

### Training the model

To train the model, run the following command:

```Python
train.py
```

This script will load the dataset, preprocess the data, build the model, and train it. The trained model will be saved as model.h5.

### Evaluating the Model

To evaluate the model on the test dataset, use:

```Python
evaluate.py
```

This script will load the saved model and output the accuracy and loss on the test dataset.

### Visualizing Results

To visualize the results, including training history and example predictions, run:

```Python
visualize.py
```

## Results

The model achieves an accuracy of approximately XX% on the test set. Below is a confusion matrix of the model's performance:

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* The book *Neural Networks from Scratch in Python* by Harrison Kinsley & Daniel Kukieła
* The Fashion MNIST dataset was created by Zalando Research.
