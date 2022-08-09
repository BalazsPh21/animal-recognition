import sys
import tensorflow as tf
import numpy as np
import cv2

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python animal.py data_directory")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets



def load_data(data_dir):
    raise NotImplementedError