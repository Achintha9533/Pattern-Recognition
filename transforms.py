import torchvision.transforms as T

def get_transform(image_size):
    """
    Returns the image preprocessing transform for CT images.

    Args:
        image_size (tuple): Desired output image size (height, width).

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    transform_post_hu = T.Compose([
        T.ToPILImage(),               # Convert numpy array to PIL Image for transformations
        T.Resize(image_size),         # Resize to specified size
        T.ToTensor(),                 # Convert PIL Image to torch tensor (C x H x W), scales [0,1]
        # Normalization to [-1, 1] happens within load_dicom_image after HU windowing.
    ])
    return transform_post_hu