import os
import matplotlib.pyplot as plt
from PIL import Image as PilImage
import numpy as np


def get_path_from_id(disk_label: str, id: int, modality: str):
    """
    Returns the full path to a file according to specific rules.

    The ruleset used for generating the full path is as follows:
    - disk_image is the full path to the folder where all of the files
        are located.
    - the following part is of format [conversion]_[modality]. This way, each 
        medical imaging modality is its own thing.
    - the next part of the path consists of the id, excluding the last three
        digits.
    - a delimiter is placed before the next step
    - the full ID is placed here, and afterwards a dot with a dcm extension ('.dcm').
        This extension means that the generated PNGs were extracted from the dicom file of the same name.

    IMPORTANT: This function does not check if the generated
    file path is actually valid and that the file exists on
    filesystem.

    ALSO IMPORTANT: The returned value is a path to a DIRECTORY!
    This directory can contain one or multiple images, all of which were extracted
    from the original DICOM file. A DICOM file is often a 3D volume, hence
    it can be split into multiple 2D PNGs.

    Args:
        disk_label: location of the folder where all of the files \
            are located (Tramontana, local storage, etc.)

        id (`int`): unique identifier of the DICOM object.

        modality(str): capturing modality of the image. Used if extension
            is `'png'`, because then the images are saved on a slightly different
            path which is related to their imaging modality.
            Defaults to None.

    Returns:
        A string, the full path to the destination directory.
    """
    path = os.path.join(
        disk_label,
        f'conversion_{modality}',
        str(id)[:-3],
        # the extension here is deliberately set to dcm
        # because of the ExportDcm rules!
        f'{modality}_{id}.dcm'
    )
    return path

def get_image_from_path(path:str):
    """
    Load the image from directory located on *path*.
    This will return an array of PIL Images.
    """
    # the "path" arg is actually the directory name where we can find
    # all image slices
    slices = []
    for item in os.scandir(path):
        _img_slice = PilImage.open(fp = item.path)
        slices.append(_img_slice)
    return slices

def plot_mosaic_of_image_slices(slices):
    """Just a handy method for plotting a mosaic of image slices."""
    slices_cnt = len(slices)
    mosaic_w = int(np.ceil(np.sqrt(slices_cnt)))
    mosaic_h = int(np.ceil(slices_cnt / mosaic_w))

    fig, axes = plt.subplots(nrows=mosaic_h, ncols=mosaic_w, tight_layout=True, figsize=(1.7*mosaic_w, 1.7*mosaic_h))

    axes_iterable = axes.flat if \
        isinstance(axes, np.ndarray) else np.array([axes])

    for img_slice, ax in zip(slices, axes_iterable):
        ax.imshow(img_slice, cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()
    plt.close(fig)

