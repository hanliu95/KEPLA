comet_support = True
try:
    from comet_ml import Experiment
except ImportError:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False

from models import KEPLA
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer

import torch
import argparse
import warnings
import os
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="KEPLA for PLA prediction")
parser.add_argument("--cfg", required=True, help="path to config file", type=str)
parser.add_argument("--data", required=True, type=str, metavar="DATA", help="dataset")
parser.add_argument(
    "--split",
    default="random",
    type=str,
    metavar="S",
    help="split type",
    # choices=["random", "cold", "cluster"],
)

args = parser.parse_args()


def build_dataloader(dataset, cfg, shuffle=False, drop_last=False):
    kwargs = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": drop_last,
        "collate_fn": graph_collate_func,
        "pin_memory": device.type == "cuda",
    }

    if cfg.SOLVER.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)


def load_datasets(data_folder):

    train_path = os.path.join(data_folder, "train.csv")
    val_path = os.path.join(data_folder, "val.csv")
    test_path = os.path.join(data_folder, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Cannot find train file: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Cannot find validation file: {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Cannot find test file: {test_path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)

    return train_dataset, val_dataset, test_dataset


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    set_seed(cfg.SOLVER.SEED)

    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)

    experiment = None

    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    data_folder = os.path.join("./datasets", args.data, args.split)

    train_dataset, val_dataset, test_dataset = load_datasets(data_folder)

    train_generator = build_dataloader(
        train_dataset,
        cfg,
        shuffle=True,
        drop_last=False,
    )

    val_generator = build_dataloader(
        val_dataset,
        cfg,
        shuffle=False,
        drop_last=False,
    )

    test_generator = build_dataloader(
        test_dataset,
        cfg,
        shuffle=False,
        drop_last=False,
    )

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False,
        )

        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Batch_size": cfg.SOLVER.BATCH_SIZE,
            "Seed": cfg.SOLVER.SEED,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "Data": args.data,
            "Split": args.split,
        }

        experiment.log_parameters(hyper_params)

        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)

        experiment.set_name(f"{args.data}_{args.split}_{suffix}")

    model = KEPLA(**cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.SOLVER.LR,
        weight_decay=getattr(cfg.SOLVER, "WEIGHT_DECAY", 1e-4),
    )

    torch.backends.cudnn.benchmark = True

    trainer = Trainer(
        model,
        opt,
        device,
        train_generator,
        val_generator,
        test_generator,
        opt_da=None,
        discriminator=None,
        experiment=experiment,
        **cfg,
    )

    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == "__main__":
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
