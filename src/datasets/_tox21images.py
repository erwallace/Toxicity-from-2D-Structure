from PIL import Image
from torchvision import transforms
from src.datasets import Tox21Base


class Tox21Images(Tox21Base):
    def __init__(self, csv_path, transform=None):
        super().__init__(csv_path, transform)
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path)
        if self.image_transform:
            image = self.image_transform(image)
        target = self.data.iloc[idx, 1]
        sample = (image, target)

        if self.transform:
            sample = self.transform(sample)

        return sample
