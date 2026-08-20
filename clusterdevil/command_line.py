import argparse

from clusterdevil.clusterdevil import cluster

parser = argparse.ArgumentParser()

parser.add_argument("model", type=str, help="Either a model architecture like [rt-detr, detr, mae] or a path to a training checkpoint")
parser.add_argument("data", type=str, help="Path to a dataset (root directory or yaml file). Multiple space-separated datasets are supported", nargs="+")

def main():
    import sys
    exit_code = dispatch_command(sys.argv[1:])
    sys.exit(exit_code)

def dispatch_command(argv: list[str]) -> int:
    args = parser.parse_args(argv)
    cluster(args.model, args.data)
    return 0
