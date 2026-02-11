import os, sys
from torchvision import transforms
from torchvision.datasets import CelebA


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor()
    ]
)


class CustomCeleb(CelebA):
    def __init__(self, root="~/data", split='train', download=True, transform=transform):
        super().__init__(root=root, split=split, download=download, transform=transform, target_type = ['identity'])
        self.attr_names = self.attr_names[:-1]

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        if self.args.task == "reconstruction":
            _target = img
        elif self.args.task == "identity":
            _target = target
        else:
            raise ValueError(f"There is no task {self.args.task} in CelebA")

        return img, _target