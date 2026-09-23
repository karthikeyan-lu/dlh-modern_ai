#!/usr/bin/env python3
""" Module for unfreezing the top layers of a pretrained base model """


def unfreeze_top_layers(model, n_layers):
    """ Unfreeze the last n_layers of a base model,
            leaving earlier layers frozen.

    Args:
        model : The base model whose layers will be unfrozen.
        n_layers : Number of last layers to set as trainable.

    Returns:
        None
    """

    if n_layers <= 0:
        for layer in model.layers:
            layer.trainable = False
        return None

    split_index = max(0, len(model.layers) - n_layers)

    for layer in model.layers[:split_index]:
        layer.trainable = False

    for layer in model.layers[split_index:]:
        layer.trainable = True

    return None
