import os


def get_image_dir(images_root: str, image_id: int, modality: str) -> str:
    return os.path.join(
        images_root,
        f'conversion_{modality}',
        str(image_id)[:-3],
        f'{modality}_{image_id}.dcm',
    )


def list_slices(images_root: str, image_id: int, modality: str) -> list[str]:
    # sorted because os.listdir order isn't guaranteed - eval needs the same slice every run
    folder = get_image_dir(images_root, image_id, modality)
    return sorted(f for f in os.listdir(folder) if f.lower().endswith('.png'))
