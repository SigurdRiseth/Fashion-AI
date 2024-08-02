import cv2
import numpy as np

from src.model.model import Model

# Load trained model
model = Model.load("../fashion_mnist.model")

# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def preprocess_image(image_path):
    """
    Preprocess the image to match the format expected by the model.
    """
    # Read image data as grayscale
    image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    # Invert colors to match dataset
    image_data = 255 - image_data

    # Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    return image_data


def predict_image(image_path):
    """
    Predict the label of the given image.
    """
    image_data = preprocess_image(image_path)

    # Predict on the image
    predictions = model.predict(image_data)

    # Get prediction instead of confidence levels
    predicted_label_index = model.output_layer_activation.predictions(predictions)[0]

    # Get label name from label index
    predicted_label_name = fashion_mnist_labels[predicted_label_index]

    return predicted_label_name


# Paths to the images
image_paths = {
    'T-shirt': '../custom_images/tshirt.png',
    'Trouser': '../custom_images/pants.png'
}

# Predict and print results for each image
for label, path in image_paths.items():
    prediction = predict_image(path)
    print(f"{label} was predicted as: {prediction}")
