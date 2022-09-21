"""
This is the main command-line interface for performing optimisation on a CNN
model. Example usage is shown below:

```shell
usage: python -m samo [-h] --model PATH -b {fpgaconvnet,finn,hls4ml} -p PATH
        -o PATH [--optimiser {brute,annealing,init,rule}]
        [--objective {throughput,latency}] [--enable_reconf {true,false}]
        [--seed N] [--batch-size N]

optional arguments:
  -h, --help            show this help message and exit
  -m PATH, --model PATH
                        path to the CNN model that you wish to
                        optimise (.keras, .onnx)
  -b {fpgaconvnet,finn,hls4ml}, --backend {fpgaconvnet,finn,hls4ml}
                        target backend for accelerating the model
  -p PATH, --platform PATH
                        hardware platform details (.json)
  -o PATH, --output-path PATH
                        output path for the optimised model (.json, .onnx)
  --optimiser {brute,annealing,init,rule}
                        optimiser to use
  --objective {throughput,latency}
                        Optimiser objective
  --enable_reconf {true,false}
                        multiple partitions
  --seed N              Seed for the optimiser run
  --batch-size N        batch size
```
"""

import logging
import argparse
import importlib
import json
import copy
import time
import random
import sys

import numpy as np

from samo.optimiser.annealing import SimulatedAnnealing
from samo.optimiser.rule import RuleBased
from samo.optimiser.brute import BruteForce

def main(args):

    # logging configuration
    logging.basicConfig(filename='samo.log', filemode='w', level=logging.INFO)

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
    parser.add_argument('--objective', choices=['throughput','latency'], required=False, default="latency", help='Optimiser objective')
    parser.add_argument("--enable_reconf", choices=["true", "false"], required=False, default="true", help="multiple partitions")
    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for the optimiser run')
    parser.add_argument('--batch-size', metavar='N', type=int, default=256,
        help='batch size')

    args = parser.parse_args(args)

    # get the batch size
    batch_size = args.batch_size
    if args.objective == "latency":
        batch_size = 1

    # print the run setting
    print("#### ( run settings ) ####")
    print(f" * model    : {args.model}")
    print(f" * backend  : {args.backend}")
    print(f" * platform : {args.platform}")
    print(f" * optimiser : {args.optimiser}")
    print(f" * objective : {args.objective}")
    print(f" * batch size : {args.batch_size}")
    print(f" * enable_reconf : {args.enable_reconf}")
    print(f" * seed : {args.seed}")
    print(f" * output_path : {args.output_path}")
    print("##########################")

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)


    # get the backend parser and exporter
    parser = importlib.import_module(f"samo.backend.{args.backend}.parser")
    exporter = importlib.import_module(f"samo.backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model, platform, batch_size)

    graph.enable_reconf = {"true":True, "false":False}[args.enable_reconf]
    graph.objective = args.objective

    random.seed(args.seed)
    np.random.seed(args.seed)

    # init
    for partition in graph.partitions:
        partition.reset()

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "rule":
        opt = RuleBased(graph)
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    opt.start_time = time.time()

    # split up the network completely
    can_split = args.optimiser != "brute"
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                can_split = True
                prev = opt.network.check_constraints()
                opt.network.split(i, valid_splits[0])
                if prev and not opt.network.check_constraints():
                    can_split = False
                    opt.network = network_copy

    # validate generated design
    assert opt.network.check_constraints(), "Intial design infeasible!"

    # run the optimiser
    opt.optimise()

    # validate generated design
    assert opt.network.check_constraints(), "Optimised design infeasible!"

    # print a summary of the run
    opt.network.summary()

    # export the design
    exporter.export(opt.network, args.model, args.output_path)

if __name__ == "__main__":
    main(sys.argv[1:])
