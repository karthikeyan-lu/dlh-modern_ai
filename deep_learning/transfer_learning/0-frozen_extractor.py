#!/usr/bin/env python3
""" Frozen pretrained CNN feature extractor. """
from tensorflow import keras


def build_feature_extractor():
    """ Build a frozen MobileNetV2 feature extractor.

    Returns:
        keras.Model: A model that maps (224, 224, 3) input images
        to 1280-dimensional feature vectors.
    """
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    return keras.Model(inputs=inputs, outputs=x)
