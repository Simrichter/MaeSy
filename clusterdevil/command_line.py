import argparse

from clusterdevil.clusterdevil import cluster

parser = argparse.ArgumentParser()

parser.add_argument("model", type=str, help="Either a model architecture like [rt-detr, detr, mae] or a path to a training checkpoint")
parser.add_argument("data", type=str, help="Path to a dataset (root directory or yaml file). Multiple space-separated datasets are supported", nargs="+")
parser.add_argument("faiss_similarity_threshold", type=float, help="Largest acceptable similarity threshold for FAISS clustering. Lower values result in less images.", default=0.75)
parser.add_argument("--intermediate-layer", type=str, help="Only for ONNX models: Name of intermediate layer after which features are taken", default="")

def main():
    import sys
    exit_code = dispatch_command(sys.argv[1:])
    sys.exit(exit_code)

def dispatch_command(argv: list[str]) -> int:
    args = parser.parse_args(argv)
    cluster(args.model, args.data, faiss_threshold = args.faiss_similarity_threshold, intermediate_layer=None if args.intermediate_layer == "" else args.intermediate_layer)
    return 0
