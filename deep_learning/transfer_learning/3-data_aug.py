#!/usr/bin/env python3
""" Data augmentation module for image preprocessing. """

from tensorflow import keras


def build_data_augmentation():
    """ Create a Keras Sequential model for image data augmentation.

    Returns:
        keras.Sequential: Augmentation pipeline with reproducible seeding.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(0.15, seed=42),
        keras.layers.RandomZoom(0.15, seed=42),
        keras.layers.RandomContrast(0.1, seed=42),
    ])
