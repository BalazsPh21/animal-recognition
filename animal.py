import sys
from matplotlib import image
import tensorflow as tf
import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 1000
IMG_HEIGHT = 1000
NUM_CATEGORIES = 10
TEST_SIZE = 0.4

CATEGORIZE = {
    "butterfly": 0,
    "cat": 1,
    "chicken": 2,
    "cow": 3,
    "dog": 4,
    "elephant": 5,
    "horse": 6,
    "sheep": 7,
    "spider": 8,
    "squirrel": 9
}

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python animal.py data_directory")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training set
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate NN performance
    model.evaluate(x_test, y_test, verbose=2)


def load_data(data_dir):
    images = list()
    labels = list()

    for animal in os.listdir(data_dir):
        folder = os.path.join(data_dir, animal)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

            images.append(img)
            labels.append(CATEGORIZE[animal])

    return images, labels


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="sigmoid", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="sigmoid"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()