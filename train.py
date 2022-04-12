from pathlib import Path
import tensorflow.keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data import load_dataset

import numpy as np


# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")


DATA_DIRECTORY = Path().absolute()  # use current working directory
TRAINING_OUTPUT_DIRECTORY = Path().absolute()


# TODO weight file has to be given to students... 500 MB file
PRETRAINED_VGG16_WEIGHTS_FILE = (
    DATA_DIRECTORY
    / "pretrained_weights"
    / "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
)
CHECKPOINT_MODEL_FILE = TRAINING_OUTPUT_DIRECTORY / "vgg16_best_val_accuracy.h5"


# Load dataset
full_dataset = load_dataset(
    input_size=224, new_spacing_mm=0.2, generate_if_not_present=True
)
training_inputs = full_dataset["inputs"]
training_labels = full_dataset["labels"]
print(f"Finished loading training data... {training_inputs.shape}")
# TODO add stratified validation / training split

# Split dataset into two data generators for training and validation
# Technically we could directly pass the data into the fit function of the model
# But using generators allows for a simple way to add augmentations and preprocessing

# This method can be used to implement custom preprocessing/augmentations during training
def custom_augmentation_fn(data: np.ndarray):
    return data


base_data_generator = ImageDataGenerator(
    validation_split=0.2,  # 20% of data is reserved for validation
    preprocessing_function=custom_augmentation_fn,
    # More options can be found in the keras docs...
)
data_generator_params = dict(
    batch_size=32, sample_weight=None, seed=None, save_to_dir=None,
)
training_data_generator = base_data_generator.flow(
    training_inputs,
    training_labels,
    shuffle=True,
    subset="training",
    **data_generator_params,
)
validation_data_generator = base_data_generator.flow(
    training_inputs,
    training_labels,
    shuffle=False,
    subset="validation",
    **data_generator_params,
)


# We use the VGG16 model
model = VGG16(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)

# Show the model layers
print(model.summary())

# Load the pretrained imagenet VGG model weights except for the last layer
# Because the pretrained weights will have a data size mismatch in the last layer of our model
# two warnings will be raised, but these can be safely ignored.
model.load_weights(str(PRETRAINED_VGG16_WEIGHTS_FILE), by_name=True, skip_mismatch=True)

# Prepare model for training by defining the loss, optimizer, and metrics to use
model.compile(
    optimizer=SGD(learning_rate=0.0001),
    loss=categorical_crossentropy,
    metrics=["accuracy"],
)

# Start actual training process
callbacks = [
    TerminateOnNaN(),
    ModelCheckpoint(
        str(CHECKPOINT_MODEL_FILE),
        monitor="val_accuracy",
        mode="auto",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
    ),
    EarlyStopping(
        monitor="val_accuracy", mode="auto", min_delta=0, patience=20, verbose=1,
    ),
]
history = model.fit(
    training_data_generator,
    steps_per_epoch=len(training_data_generator),
    validation_data=validation_data_generator,
    validation_steps=None,
    validation_freq=1,
    epochs=100,
    callbacks=callbacks,
    verbose=2,
)


# TODO visualize/plot some of the training history...
print(history)

# TODO load and use the trained model for prediction...
