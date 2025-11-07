# TODO

- [ ] Fichier `presets` : implémentation comme décrit dans `training/TODO.md`
    - Traduire les presets en objets/ dicts réutilisables
    - Charger/valider les presets depuis fichier YAML ou JSON

- [ ] Fichiers pour loader un modèle depuis la config avec ces params
    - `model.from_config(config)` : architecture, pretrained, device, checkpoint_path
    - Gestion des dépendances / versions des poids

- [ ] Ajouter les metrics dans le `.txt` lors de la sauvegarde + sauvegarde du best
    - Écrire fichier `metrics.txt` par checkpoint (epoch, train/val loss, métriques clé)
    - Maintenir `best` checkpoint selon métrique configurée (ex: val_accuracy)
    - Log succinct pour chaque sauvegarde