from pathlib import Path
from keras.utils import data_utils


def maybe_download_vgg16_pretrained_weights(weights_file: Path):
    if not weights_file.is_file():
        print(f"Pretrained VGG16 weigths not found, downloading now to: {weights_file}")
        weights_file.parent.mkdir(parents=True, exist_ok=True)
        data_utils.get_file(
            "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
            (
                "https://storage.googleapis.com/tensorflow/keras-applications/"
                "vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
            ),
            cache_subdir=str(weights_file.parent.absolute()),
            file_hash="64373286793e3c8b2b4e3219cbf3544b",
        )
