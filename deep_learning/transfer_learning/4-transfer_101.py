#!/usr/bin/env python3
""" Transfer learning on the Caltech-101 dataset (101 classes+background). """
import tensorflow as tf
from tensorflow import keras


def train_transfer_model():
    """ Build, train, and save a transfer-learning classifier for Caltech-101.

    The pipeline:
      1. Loads MobileNetV2 (ImageNet weights, no top) as a frozen backbone.
      2. Applies data augmentation + model-specific preprocessing.
      3. Trains a custom classification head (base frozen).
      4. Fine-tunes the top 30 layers of the backbone with a low LR.
      5. Saves the best model to ``caltech101_model.h5``.

    Returns:
        keras.Model: The best trained model.
    """
    DATA_DIR = "101_ObjectCategories"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    SEED = 42
    INITIAL_EPOCHS = 10
    FINETUNE_EPOCHS = 15

    # ------------------------------------------------------------------ #
    # 1. Datasets                                                        #
    # ------------------------------------------------------------------ #
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    num_classes = len(train_ds.class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # ------------------------------------------------------------------ #
    # 2. Data augmentation                                               #
    # ------------------------------------------------------------------ #
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=SEED),
            keras.layers.RandomRotation(0.15, seed=SEED),
            keras.layers.RandomZoom(0.15, seed=SEED),
            keras.layers.RandomContrast(0.1, seed=SEED),
        ],
        name="data_augmentation",
    )

    # ------------------------------------------------------------------ #
    # 3. Pretrained backbone (frozen)                                    #
    # ------------------------------------------------------------------ #
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # ------------------------------------------------------------------ #
    # 4. Build full model                                                #
    # ------------------------------------------------------------------ #
    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            baseline=0.85,
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
        keras.callbacks.ModelCheckpoint(
            "caltech101_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    # ------------------------------------------------------------------ #
    # 5. Phase 1 — train the classification head only                    #
    # ------------------------------------------------------------------ #
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------ #
    # 6. Phase 2 — fine-tune the top layers of the backbone              #
    # ------------------------------------------------------------------ #
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks,
    )

    # Return the best checkpoint (highest val_accuracy) saved during training.
    best_model = keras.models.load_model("caltech101_model.h5")
    return best_model


if __name__ == "__main__":
    train_transfer_model()
