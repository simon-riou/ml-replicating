# TODO
"""

1. train.py (Point d'entrée)

Ce fichier sera l'équivalent de la fonction main(args) et get_args_parser() de votre fichier train.py. Il doit :

    Parser les arguments (similaire à get_args_parser).

    Charger la configuration (idéalement à partir d'un fichier dans /configs).

    Initialiser le DDP (si distribué, comme utils.init_distributed_mode).

    Charger les données (data_loader, data_loader_test via /datasets).

    Créer le modèle (torchvision.models.get_model ou une de vos classes dans /models).

    Initialiser l'Optimiseur/Scheduler (via /utils/optimizer.py et /training/lr_schedulers.py).

    Lancer la boucle d'entraînement (via training/trainer.py).
    
"""