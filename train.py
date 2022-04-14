from pathlib import Path
from typing import Tuple
from enum import Enum, unique

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

from balanced_sampler import sample_balanced, UndersamplingIterator
from data import load_dataset
from utils import maybe_download_vgg16_pretrained_weights


# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")


# This should point at the directory containing the source LUNA22 prequel dataset
DATA_DIRECTORY = Path().absolute() / "LUNA22 prequel"

# This should point at a directory to put the preprocessed/generated datasets from the source data
GENERATED_DATA_DIRECTORY = Path().absolute()

# This should point at a directory to store the training output files
TRAINING_OUTPUT_DIRECTORY = Path().absolute()

# This should point at the pretrained model weights file for the VGG16 model.
# The file can be downloaded here:
# https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
PRETRAINED_VGG16_WEIGHTS_FILE = (
    Path().absolute()
    / "pretrained_weights"
    / "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
)
maybe_download_vgg16_pretrained_weights(PRETRAINED_VGG16_WEIGHTS_FILE)


# Load dataset
# This method will generate a preprocessed dataset from the source data if it is not present (only needs to be done once)
# Otherwise it will quickly load the generated dataset from disk
full_dataset = load_dataset(
    input_size=224,
    new_spacing_mm=0.2,
    cross_slices_only=True,
    generate_if_not_present=True,
    always_generate=False,
    source_data_dir=DATA_DIRECTORY,
    generated_data_dir=GENERATED_DATA_DIRECTORY,
)
inputs = full_dataset["inputs"]


@unique
class MLProblem(Enum):
    malignancy_prediction = "malignancy"
    nodule_type_prediction = "noduletype"


# Here you can switch the machine learning problem to solve
problem = MLProblem.malignancy_prediction

# Configure problem specific parameters
if problem == MLProblem.malignancy_prediction:
    # We made this problem a binary classification problem:
    # 0 - benign, 1 - malignant
    num_classes = 2
    batch_size = 32
    # Take approx. 15% of all samples for the validation set and ensure it is a multiple of the batch size
    num_validation_samples = int(len(inputs) * 0.15 / batch_size) * batch_size
    labels = full_dataset["labels_malignancy"]
    # It is possible to generate training labels yourself using the raw annotations of the radiologists...
    labels_raw = full_dataset["labels_malignancy_raw"]
elif problem == MLProblem.nodule_type_prediction:
    # We made this problem a multiclass classification problem with three classes:
    # 0 - non-solid, 1 - part-solid, 2 - solid
    num_classes = 3
    batch_size = 30  # make this a factor of three to fit three classes evenly per batch during training
    # This dataset has only few part-solid nodules in the dataset, so we make a tiny validation set
    num_validation_samples = batch_size * 2
    labels = full_dataset["labels_nodule_type"]
    # It is possible to generate training labels yourself using the raw annotations of the radiologists...
    labels_raw = full_dataset["labels_nodule_type_raw"]
else:
    raise NotImplementedError(f"An unknown MLProblem was specified: {problem}")

print(
    f"Finished loading data for MLProblem: {problem}... X:{inputs.shape} Y:{labels.shape}"
)

# partition small and class balanced validation set from all data
validation_indices = sample_balanced(
    input_labels=np.argmax(labels, axis=1),
    required_samples=num_validation_samples,
    class_balance=None,  # By default sample with equal probability, e.g. for two classes : {0: 0.5, 1: 0.5}
    shuffle=True,
)
validation_mask = np.isin(np.arange(len(labels)), list(validation_indices.values()))
training_inputs = inputs[~validation_mask, :]
training_labels = labels[~validation_mask, :]
validation_inputs = inputs[validation_mask, :]
validation_labels = labels[validation_mask, :]

print(f"Splitted data into training and validation sets:")
training_class_counts = np.unique(
    np.argmax(training_labels, axis=1), return_counts=True
)[1]
validation_class_counts = np.unique(
    np.argmax(validation_labels, axis=1), return_counts=True
)[1]
print(
    f"Training   set: {training_inputs.shape} {training_labels.shape} {training_class_counts}"
)
print(
    f"Validation set: {validation_inputs.shape} {validation_labels.shape} {validation_class_counts}"
)


# Split dataset into two data generators for training and validation
# Technically we could directly pass the data into the fit function of the model
# But using generators allows for a simple way to add augmentations and preprocessing
# It also allows us to balance the batches per class using undersampling

# The following methods can be used to implement custom preprocessing/augmentation during training


def clip_and_scale(
    data: np.ndarray, min_value: float = -1000.0, max_value: float = 400.0
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data


def random_flip_augmentation(
    input_sample: np.ndarray, axis: Tuple[int, ...] = (1, 2)
) -> np.ndarray:
    for ax in axis:
        if np.random.random_sample() > 0.5:
            input_sample = np.flip(input_sample, axis=ax)
    return input_sample


def shared_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    """Preprocessing that is used by both the training and validation sets during training

    :param input_batch: np.ndarray [batch_size x channels x dim_x x dim_y]
    :return: np.ndarray preprocessed batch
    """
    input_batch = clip_and_scale(input_batch, min_value=-1000.0, max_value=400.0)
    # Can add more preprocessing here...
    return input_batch


def train_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)

    output_batch = []
    for sample in input_batch:
        sample = random_flip_augmentation(sample, axis=(1, 2))
        output_batch.append(sample)

    return np.array(output_batch)


def validation_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)
    return input_batch


training_data_generator = UndersamplingIterator(
    training_inputs,
    training_labels,
    shuffle=True,
    preprocess_fn=train_preprocess_fn,
    batch_size=batch_size,
)
validation_data_generator = UndersamplingIterator(
    validation_inputs,
    validation_labels,
    shuffle=False,
    preprocess_fn=validation_preprocess_fn,
    batch_size=batch_size,
)


# We use the VGG16 model
model = VGG16(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=num_classes,
    classifier_activation="softmax",
)

# Show the model layers
print(model.summary())

# Load the pretrained imagenet VGG model weights except for the last layer
# Because the pretrained weights will have a data size mismatch in the last layer of our model
# two warnings will be raised, but these can be safely ignored.
model.load_weights(str(PRETRAINED_VGG16_WEIGHTS_FILE), by_name=True, skip_mismatch=True)

# Prepare model for training by defining the loss, optimizer, and metrics to use
# Output labels and predictions are one-hot encoded, so we use the categorical_accuracy metric
model.compile(
    optimizer=SGD(learning_rate=0.0001, momentum=0.8, nesterov=True),
    loss=categorical_crossentropy,
    metrics=["categorical_accuracy"],
)

# Start actual training process
output_model_file = (
    TRAINING_OUTPUT_DIRECTORY / f"vgg16_{problem.value}_best_val_accuracy.h5"
)
callbacks = [
    TerminateOnNaN(),
    ModelCheckpoint(
        str(output_model_file),
        monitor="val_categorical_accuracy",
        mode="auto",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
    ),
    EarlyStopping(
        monitor="val_categorical_accuracy",
        mode="auto",
        min_delta=0,
        patience=100,
        verbose=1,
    ),
]
history = model.fit(
    training_data_generator,
    steps_per_epoch=len(training_data_generator),
    validation_data=validation_data_generator,
    validation_steps=None,
    validation_freq=1,
    epochs=250,
    callbacks=callbacks,
    verbose=2,
)


# generate a plot using the training history...
output_history_img_file = (
    TRAINING_OUTPUT_DIRECTORY / f"vgg16_{problem.value}_train_plot.png"
)
print(f"Saving training plot to: {output_history_img_file}")
plt.plot(history.history["categorical_accuracy"])
plt.plot(history.history["val_categorical_accuracy"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.savefig(str(output_history_img_file), bbox_inches="tight")
