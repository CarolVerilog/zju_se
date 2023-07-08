import argparse
from gui import GUI

parser = argparse.ArgumentParser()
parser.add_argument(
    "--width",
    type=int,
    default=800,
    help="window width",
)
parser.add_argument(
    "--height",
    type=int,
    default=600,
    help="window height",
)
args = parser.parse_args()

gui = GUI(args.width, args.height)
gui.render()
