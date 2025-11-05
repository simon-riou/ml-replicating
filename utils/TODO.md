├── utils/                   # Fonctions utilitaires réutilisables
│   ├── __init__.py
│   ├── metrics.py           # Fonctions comme accuracy, calculate_fid (similaire à utils.accuracy)
│   ├── distributed.py       # Fonctions pour DDP, synchronisation (similaire à utils.init_distributed_mode)
│   ├── optimizer.py         # Fonction pour configurer l'optimiseur et le weight decay
│   └── checkpoints.py       # Fonctions de sauvegarde/chargement (similaire à utils.save_on_master)