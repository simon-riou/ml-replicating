# Paper replicating

This repo is a collection of model or technique from machine learning and deep learning from the papers I studied, those papers can be found here https://simon-riou.github.io/ml-reading/

# Example

To train a model run one of the following command:

    - train.py --config [config_path/config.yaml]   (To train a model from a given config, with default settings)
    - train.py --config [config_path/config.yaml] --epochs X --...   (To train a model from a given config, with default settings if not included in config and with ovewritten settings)
    - train.py --resume [model_path/model.pkg] --config [config_path/config.yaml]   (To keep training a model from the given settings)
    - train.py --resume [model_path/model.pkg] --config [config_path/config.yaml] --epochs X --start-epoch Y --...   (To keep training a model with the default settings and overwritten settings)

# Config

The config file MUST include the given parameters:
    - model with a type
    - dataset with a type
    - optimizer with a type
    - criterion with a type