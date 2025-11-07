├── datasets/                # Contient les fonctions pour charger et prétraiter les données
│   ├── __init__.py
│   ├── classification_data.py # Fonctions pour ImageFolder, CIFAR10, etc. (similaire à load_data dans train.py)
│   └── diffusion_data.py


3. /datasets (Gestion des Données)

Le fichier load_data de votre train.py doit être décomposé.

    classification_data.py : Contient les fonctions pour créer les Dataset (comme torchvision.datasets.ImageFolder ou CIFAR10) et les DataLoader pour la classification.

        Utilisez des presets de transformation (comme presets.ClassificationPresetTrain dans train.py) définis dans training/presets.py.

    diffusion_data.py : S'occuperait spécifiquement des jeux de données pour DDPM.