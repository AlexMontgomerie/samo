import argparse
import importlib

from optimiser.annealing import SimulatedAnnealing
from optimiser.brute import BruteForce

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    args = parser.parse_args()

    # get the correct backend parser
    parser = importlib.import_module(f"backend.{args.backend}.parser")

    # parse the example network
    graph = parser.parse(args.model)

    # platform
    platform = {
        "LUT" : 437200,
        "DSP" : 900,
        "BRAM" : 1090,
        "FF" : 218600
    }

    # perform optimisation on the computation graph
    opt = SimulatedAnnealing(graph)

    opt.network.platform = platform

    print("latency (before): ", opt.network.eval_latency())
    opt.optimise()

    print("latency (after): ", opt.network.eval_latency())


if __name__ == "__main__":
    main()
