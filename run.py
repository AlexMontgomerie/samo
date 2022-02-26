import argparse
import importlib
import json

from optimiser.annealing import SimulatedAnnealing
from optimiser.rule import RuleBased
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
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for the optimised model (.json, .onnx)")
    parser.add_argument("--optimiser", choices=["brute", "annealing", "init", "rule"], required=False, default="annealing",
            help="optimiser to use")
    parser.add_argument("--enable_reconf", choices=["true", "false"], required=False, default="false", help="multiple partitions")

    args = parser.parse_args()

    # get the backend parser and exporter
    parser = importlib.import_module(f"backend.{args.backend}.parser")
    exporter = importlib.import_module(f"backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model)

    graph.enable_reconf = {"true":True, "false":False}[args.enable_reconf]

    # init
    for partition in graph.partitions:
        partition.reset()

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "rule":
        opt = RuleBased(graph)
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        #graph.save_config("cnv_finn_baseline.json")
        #graph.load_config("configs/mobilenetv1_finn_baseline.json")

        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    # update the platform resource constraints
    opt.network.load_platform(platform)

    # split up the network completely
    can_split = True
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            if valid_splits:
                can_split = True
                opt.network.split(i, valid_splits[0])

    # run the optimiser
    # opt.optimise()

    # validate generated design
    assert(opt.network.check_constraints())

    # print a summary of the run
    opt.network.summary()

    # export the design
    exporter.export(opt.network, args.model, args.output_path)

if __name__ == "__main__":
    main()
