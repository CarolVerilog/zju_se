import argparse
import pathlib

from nerfacto import NeRFacto
from gui import GUI

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    help="which scene to use",
)

args = parser.parse_args()
nerf = NeRFacto(args.data_root, args.train_split, args.scene)
gui = GUI(nerf, 800, 600)
gui.train()
