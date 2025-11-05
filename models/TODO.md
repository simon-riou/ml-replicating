├── models/                  # Contient les définitions de tous les modèles (ViT, AlexNet, etc.)
│   ├── __init__.py          # Pour importer facilement les modèles (e.g., from models import AlexNet)
│   ├── alexnet.py
│   ├── vit.py
│   ├── ddpm/                # Dossier pour les modèles de diffusion complexes
│   │   ├── unet.py
│   │   ├── ddpm.py
│   │   └── ...
│   └── ...