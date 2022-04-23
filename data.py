from enum import Enum, unique
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import SimpleITK.SimpleITK
import numpy as np
import SimpleITK


DATASET_TYPE = Dict[str, Union[List[List[int]], np.ndarray]]


@unique
class NoduleType(Enum):
    NonSolid = 1
    PartSolid = 2
    Solid = 3


@unique
class MalignancyType(Enum):
    Benign = 1
    Malignant = 2


def load_and_resample_nodule_img(
    file_name: Path, new_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    # Read image file and cast to float32
    img = SimpleITK.ReadImage(str(file_name), SimpleITK.sitkFloat32)

    # Resample image
    original_spacing_mm = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing_mm, new_spacing_mm)
    ]
    resampled_img = SimpleITK.Resample(
        img,
        new_size,
        SimpleITK.Transform(),
        SimpleITK.sitkLinear,
        img.GetOrigin(),
        new_spacing_mm,
        img.GetDirection(),
        0,
        img.GetPixelID(),
    )

    # Return image data as a numpy array
    return SimpleITK.GetArrayFromImage(resampled_img)


def get_cross_slices_from_cube(volume: np.ndarray) -> np.ndarray:
    # input should be a cube
    assert volume.shape[0] == volume.shape[1] and volume.shape[1] == volume.shape[2]
    half_size = int(volume.shape[0] // 2)
    return np.array(
        [volume[half_size, :, :], volume[:, half_size, :], volume[:, :, half_size]]
    )


def center_crop_volume(
    volume: np.ndarray,
    crop_size: np.ndarray,
    pad_if_too_small: bool = True,
    pad_value: float = -1024,
) -> np.ndarray:
    assert isinstance(crop_size, np.ndarray)
    if not all(volume.shape >= crop_size) and not pad_if_too_small:
        raise ValueError(
            f"Input volume shape ({volume.shape}) was too small to crop: {crop_size}"
        )
    else:
        # apply optimistic padding, so there always is sufficient padding before the crop
        pad_values = np.maximum(crop_size - volume.shape, np.zeros((3,))).astype(int)
        volume = np.pad(
            volume,
            pad_width=[(pad_value, pad_value) for pad_value in pad_values],
            mode="constant",
            constant_values=pad_value,
        )
        assert all(volume.shape >= crop_size)
    shp = np.floor((np.array(volume.shape) - crop_size) // 2).astype(int)
    cropped_img_data = volume[
        shp[0] : shp[0] + crop_size[0],
        shp[1] : shp[1] + crop_size[1],
        shp[2] : shp[2] + crop_size[2],
    ]
    assert all(cropped_img_data.shape == crop_size)
    return cropped_img_data


def get_label_for_texture_values(texture_values: List[int]) -> np.ndarray:
    remap_labels = {
        1: NoduleType.NonSolid.value,
        2: NoduleType.NonSolid.value,
        3: NoduleType.PartSolid.value,
        4: NoduleType.Solid.value,
        5: NoduleType.Solid.value,
    }
    output_label = {
        NoduleType.NonSolid.value: [1.0, 0.0, 0.0],
        NoduleType.PartSolid.value: [0.0, 1.0, 0.0],
        NoduleType.Solid.value: [0.0, 0.0, 1.0],
    }
    values = np.array(list(map(remap_labels.get, texture_values)))
    final_label = int(np.median(values))
    return np.array(output_label[final_label], dtype=np.float32)


def get_label_for_malignancy(malignancy_values: List[int]) -> np.ndarray:
    remap_labels = {
        1: MalignancyType.Benign.value,
        2: MalignancyType.Benign.value,
        3: MalignancyType.Malignant.value,
        4: MalignancyType.Malignant.value,
        5: MalignancyType.Malignant.value,
    }
    output_label = {
        MalignancyType.Benign.value: [1.0, 0.0],
        MalignancyType.Malignant.value: [0.0, 1.0],
    }
    values = np.array(list(map(remap_labels.get, malignancy_values)))
    final_label = int(np.median(values))
    return np.array(output_label[final_label], dtype=np.float32)


def _load_nodule_information(file_name: Path) -> List[Dict[str, Any]]:
    return np.load(str(file_name), allow_pickle=True)


def _generate_training_dataset(
    input_size: int = 224,
    new_spacing_mm: float = 0.2,
    cross_slices_only: bool = True,
    data_dir: Path = Path().absolute(),
) -> DATASET_TYPE:
    nodule_info = _load_nodule_information(file_name=data_dir / "LIDC-IDRI_1176.npy")

    # Create a dataset of [number_of_nodules x 3 x input_size x input_size] if using cross slices
    dataset_inputs = np.zeros(
        (
            len(nodule_info),
            3 if cross_slices_only else input_size,
            input_size,
            input_size,
        ),
        dtype=np.float32,
    )
    labels_malignancy = np.zeros((len(nodule_info), 2), dtype=np.float32)
    labels_nodule_type = np.zeros((len(nodule_info), 3), dtype=np.float32)
    labels_malignancy_raw = []
    labels_nodule_type_raw = []
    for i, nodule in enumerate(nodule_info):
        print(f"{i + 1}/{len(nodule_info)}")
        seriesuid = nodule["SeriesInstanceUID"]
        x, y, z = nodule["VoxelCoordX"], nodule["VoxelCoordY"], nodule["VoxelCoordZ"]
        data_filename = data_dir / "LIDC-IDRI" / f"{seriesuid}_{x}_{y}_{z}_0000.nii.gz"

        # Load the nodule data crop and resample it to have isotropic voxel spacing
        # (1 voxel corresponds to new_spacing_mm in mm).
        nodule_data = load_and_resample_nodule_img(
            file_name=data_filename,
            new_spacing_mm=(new_spacing_mm, new_spacing_mm, new_spacing_mm),
        )

        # Crop a volume of 50 mm^3 around the nodule
        nodule_data = center_crop_volume(
            volume=nodule_data,
            crop_size=np.array((input_size, input_size, input_size)),
            pad_if_too_small=True,
            pad_value=-1024,
        )

        if cross_slices_only:
            # Extract the axial/coronal/sagittal center slices of the 50 mm^3 cube
            nodule_data = get_cross_slices_from_cube(volume=nodule_data)

        # Store data
        dataset_inputs[i, :] = nodule_data
        labels_nodule_type[i, :] = get_label_for_texture_values(
            texture_values=nodule["Texture"]
        )
        labels_malignancy[i, :] = get_label_for_malignancy(
            malignancy_values=nodule["Malignancy"]
        )
        # NOTE Number of raw texture and malignancy scores can differ between nodules
        labels_nodule_type_raw.append(nodule["Texture"])
        labels_malignancy_raw.append(nodule["Malignancy"])

    return dict(
        inputs=dataset_inputs,
        labels_nodule_type=labels_nodule_type,
        labels_malignancy=labels_malignancy,
        labels_nodule_type_raw=labels_nodule_type_raw,
        labels_malignancy_raw=labels_malignancy_raw,
    )


def load_dataset(
    input_size: int = 224,
    new_spacing_mm: float = 0.2,
    cross_slices_only: bool = True,
    generated_data_dir: Path = Path().absolute(),
    source_data_dir: Path = Path().absolute(),
    generate_if_not_present: bool = True,
    always_generate: bool = False,
) -> DATASET_TYPE:
    file_name = (
        generated_data_dir
        / f"train_input_data_{input_size}_{new_spacing_mm}{'_cso' if cross_slices_only else ''}.npz"
    )
    if always_generate or (not file_name.is_file() and generate_if_not_present):
        print(
            f"Generating the preprocessed dataset file (needs to be done once): "
            f"{file_name} from the source data at: {source_data_dir}"
        )
        dataset = _generate_training_dataset(
            input_size=input_size,
            new_spacing_mm=new_spacing_mm,
            data_dir=source_data_dir,
        )
        np.savez_compressed(str(file_name), **dataset)
        return dataset
    else:
        print(f"Loading the generated preprocessed dataset file from: {file_name}")
        return np.load(str(file_name), allow_pickle=True)


if __name__ == "__main__":
    load_dataset()
