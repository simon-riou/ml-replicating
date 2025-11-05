├── training/                # Logique d'entraînement générale et schedulers
│   ├── __init__.py
│   ├── trainer.py           # Classe ou fonction principale pour la boucle train_one_epoch et evaluate
│   ├── presets.py           # Définitions de transformations/presets (similaire au fichier presets dans train.py)
│   └── lr_schedulers.py     # Définition des planificateurs de taux d'apprentissage (SequentialLR, CosineAnnealingLR)