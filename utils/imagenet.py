import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_imagenet_dataloaders(train_epochs, batch_size, num_workers, image_size=256):
    train_dataset = load_dataset("clane9/imagenet-100", split="train")
    test_dataset = load_dataset("clane9/imagenet-100", split="validation")

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def make_collate_fn(transform):
        def collate_fn(batch):
            images = [transform(item["image"].convert("RGB")) for item in batch]
            labels = [item["label"] for item in batch]
            return torch.stack(images), torch.tensor(labels)
        return collate_fn

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(test_transform),
        num_workers=num_workers // 2,
        pin_memory=True
    )

    total_train_steps = len(train_dataset) // batch_size * train_epochs

    def train_loader_gen():
        i = 0
        while True:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=make_collate_fn(train_transform),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            for batch in train_loader:
                if i >= total_train_steps:
                    return
                yield batch
                i += 1

    return train_loader_gen(), test_loader, total_train_steps