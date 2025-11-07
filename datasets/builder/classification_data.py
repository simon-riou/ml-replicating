"""
Classification dataset builder.
"""

from torchvision import datasets, transforms

def build_MNIST(root, is_train, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    dataset = datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        download=download
    )

    return dataset

def build_CIFAR10(root, is_train, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root=root,
        train=is_train,
        transform=transform,
        download=download
    )

    return dataset

def build_CIFAR100(root, is_train, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    dataset = datasets.CIFAR100(
        root=root,
        train=is_train,
        transform=transform,
        download=download
    )

    return dataset