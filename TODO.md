# TODO

- [ ] Fichier `dataset` : loader depuis la config
    - Définir interface de config (chemin, transforms, split, seed)
    - Implémenter `from_config(config)` + tests unitaires

- [ ] Fichier `presets` : implémentation comme décrit dans `training/TODO.md`
    - Traduire les presets en objets/ dicts réutilisables
    - Charger/valider les presets depuis fichier YAML ou JSON

- [ ] Modifier la sauvegarde des modèles
    - Normaliser format de checkpoint (state_dict, epoch, opt_state, metrics)
    - Ajouter rotation/retention (keep N latest)

- [ ] Fichiers pour loader un modèle depuis la config avec ces params
    - `model.from_config(config)` : architecture, pretrained, device, checkpoint_path
    - Gestion des dépendances / versions des poids

- [ ] Ajouter les metrics dans le `.txt` lors de la sauvegarde + sauvegarde du best
    - Écrire fichier `metrics.txt` par checkpoint (epoch, train/val loss, métriques clé)
    - Maintenir `best` checkpoint selon métrique configurée (ex: val_accuracy)
    - Log succinct pour chaque sauvegarde