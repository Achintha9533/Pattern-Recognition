import torchvision.transforms as T

def get_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])