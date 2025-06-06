import torchvision.transforms as T

image_size = (64, 64)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])
