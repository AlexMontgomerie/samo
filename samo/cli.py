import argparse
import importlib
import json

from samo.optimiser.annealing import SimulatedAnnealing
from samo.optimiser.brute import BruteForce

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for the optimised model (.json, .onnx)")
    parser.add_argument("--optimiser", choices=["brute", "annealing", "init"], required=False, default="annealing",
            help="optimiser to use")

    args = parser.parse_args()

    # get the backend parser and exporter
    parser = importlib.import_module(f"samo.backend.{args.backend}.parser")
    exporter = importlib.import_module(f"samo.backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model)

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    # update the platform resource constraints
    opt.network.platform = platform["resources"]

    # run the optimiser
    opt.optimise()

    # print a summary of the run
    opt.network.summary()

    # export the design
    exporter.export(opt.network, args.model, args.output_path)

if __name__ == "__main__":
    main()
