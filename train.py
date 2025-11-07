import argparse
import yaml
import sys
from pprint import pprint

from training.trainer import train

# TODO: Fix the param in argparse -> little by little
# - Add verbose

"""
The architecture was inspired by the following repos :
    - https://github.com/pytorch/vision/blob/ca2212438fdd8ce29b66999ed70ed54b0f9372d1/references/classification/train.py#L215
    - https://github.com/IgorSusmelj/pytorch-styleguide/tree/master
"""

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--config", default="", type=str, help="config path")
    #parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--num-workers", default=2, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    #parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--tb-dir", default="tb-output", type=str, help="tensorboard path to save logs")
    parser.add_argument("--run-dir", default="runs", type=str, help="path to save saved models")
    parser.add_argument(
        "--no-save", 
        action="store_true", 
        help="If set, models and checkpoints will NOT be saved."
    )
    parser.add_argument("--max-keep", default=5, type=int, help="maximum saved models")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    #parser.add_argument(
    #    "--cache-dataset",
    #    dest="cache_dataset",
    #    help="Cache the datasets for quicker initialization. It also serializes the transforms",
    #    action="store_true",
    #)
    #parser.add_argument(
    #    "--sync-bn",
    #    dest="sync_bn",
    #    help="Use sync batch norm",
    #    action="store_true",
    #)
    #parser.add_argument(
    #    "--test-only",
    #    dest="test_only",
    #    help="Only test the model",
    #    action="store_true",
    #)
    #parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    #parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    #parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    #parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    #
    ## Mixed precision training parameters
    #parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    #
    ## distributed training parameters
    #parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    #parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    #parser.add_argument(
    #    "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    #)
    #parser.add_argument(
    #    "--model-ema-steps",
    #    type=int,
    #    default=32,
    #    help="the number of iterations that controls how often to update the EMA model (default: 32)",
    #)
    #parser.add_argument(
    #    "--model-ema-decay",
    #    type=float,
    #    default=0.99998,
    #    help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    #)
    parser.add_argument(
        "--use-deterministic-algorithms", default=None, type=int, metavar="SEED", help="Forces the use of deterministic algorithms only with the given SEED."
    )
    #parser.add_argument(
    #    "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    #)
    #parser.add_argument(
    #    "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    #)
    #parser.add_argument(
    #    "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    #)
    #parser.add_argument(
    #    "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    #)
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    #parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    #parser.add_argument(
    #    "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    #)
    #parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    #parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    #parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.config:
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            parser.set_defaults(**yaml_config)
            print(f"Configuration loaded from: {args.config}")
            pprint(yaml_config)
            print()

        except FileNotFoundError:
            print(f"Error: Configuration file not found at {args.config}")
            sys.exit(1)
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            sys.exit(1)

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    train(args)