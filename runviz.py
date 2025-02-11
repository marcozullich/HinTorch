import argparse
import os

import torch

from src.architectures import (model_improved, model_mish, model_original,
                               model_relu)
from src.data import train_balanced_dl, train_imbalanced_dl
from src.init import modern_init_, original_init_
from src.loss_viz import plot_viz

MODEL_PARSER = {"original": model_original, "improved": model_improved, "relu": model_relu, "mish": model_mish}
INIT_PARSER = {"original": original_init_, "modern": modern_init_}
DATALOADER_PARSER = {"imbalanced": train_imbalanced_dl, "balanced": train_balanced_dl}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["original", "improved", "relu", "mish"], default="original")
    parser.add_argument("--init_type", type=str, choices=["original", "modern"], default="original")
    parser.add_argument("--dataloader_type", type=str, choices=["imbalanced", "balanced"], default="imbalanced")
    parser.add_argument("--loss_function", type=str, default="MSELoss")
    parser.add_argument("--device", type=str, choices=[None, "cpu", "cuda"], default=None)
    parser.add_argument("--plot_resolution", type=int, default=50)
    parser.add_argument("--save_plot", action="store_true", default=False)
    parser.add_argument("--nruns", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        loss_function = getattr(torch.nn, args.loss_function)()
    except AttributeError:
        raise ValueError(f"Loss function {args.loss_function} not found in torch.nn. Please specify a valid loss function.")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    init_fx = INIT_PARSER[args.init_type]
    model = MODEL_PARSER[args.model_type]
    init_fx(model)
    
    dataloader = DATALOADER_PARSER[args.dataloader_type]

    for i in range(args.nruns):
        save_location = f"plots/loss_viz_model:{args.model_type}_init:{args.init_type}_dl:{args.dataloader_type}_{i}.png" if args.save_plot else None
        os.makedirs(os.path.dirname(save_location), exist_ok=True)

        plot_viz(model, dataloader, loss_function, save_location, args.plot_resolution, args.device)
    



if __name__ == "__main__":
    main()