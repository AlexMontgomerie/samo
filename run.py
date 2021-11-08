import argparse
import importlib
import json

from optimiser.annealing import SimulatedAnnealing
from optimiser.brute import BruteForce

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    args = parser.parse_args()

    # get the correct backend parser
    parser = importlib.import_module(f"backend.{args.backend}.parser")

    # parse the network
    graph = parser.parse(args.model)

    # create an optimiser instance for the network
    opt = SimulatedAnnealing(graph)

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)

    # update the platform resource constraints
    opt.network.platform = platform["resources"]

    # run the optimiser
    opt.optimise()

    opt.network.summary()

if __name__ == "__main__":
    main()
