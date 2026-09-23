#!/usr/bin/env python3
""" Module for adding a classification head to a feature extractor. """
from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """ Attach a classification head to a pretrained feature extractor.

    Args:
        base_model : Model outputting a pooled feature vector.
        num_classes : Number of output classes.

    Returns:
        keras.Model: A new model ready for classification.
    """
    x = base_model.output
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=base_model.input, outputs=outputs)
