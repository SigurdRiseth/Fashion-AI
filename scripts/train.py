import os
import cv2
import numpy as np

from src.accuracy.accuracy_categorical import AccuracyCategorical
from src.activation_functions.activation_relu import ActivationReLU
from src.activation_functions.activation_softmax import ActivationSoftmax
from src.layers.layer_dense import LayerDense
from src.loss.loss_categorical_crossentropy import LossCategoricalCrossentropy
from src.model.model import Model
from src.optimizers.optimizer_adam import OptimizerAdam


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    dataset_path = os.path.join(path, dataset)
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"The path {dataset_path} is not a directory")

    labels = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            # Skip non-file entries
            if not os.path.isfile(file_path):
                continue
            # Check if the file has an acceptable image extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Read the image
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    # And append it and a label to the lists
                    X.append(image)
                    y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test


X, y, X_test, y_test = create_data_mnist('../fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
 127.5) / 127.5

# Initialize Model
model = Model()

# Add layers
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())

# Set loss, optimizer and accuracy objects
model.set(
 loss=LossCategoricalCrossentropy(),
 optimizer=OptimizerAdam(decay=1e-3),
 accuracy=AccuracyCategorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

# Save the trained model
model.save('../fashion_mnist.model')
