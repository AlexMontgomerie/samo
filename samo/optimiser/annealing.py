import csv
import math
import copy
import random
from dataclasses import dataclass, field
import time
from datetime import datetime
import json

import numpy as np
from tqdm import tqdm

from samo.model import Network

import samo.backend.fpgaconvnet.export as exporter

@dataclass
class SimulatedAnnealing:
    network: Network
    T: float = 10.0
    k: float = 100.0
    T_min: float = 0.001
    cool: float = 0.995
    iterations: int = 10
    valid_variables: list = field(default_factory=lambda: ["channel_in_folding", "channel_out_folding", "kernel_folding"])

    def update(self):
        for partition in self.network.partitions:
            for index, layer in enumerate(partition):
                partition.nodes[layer]["hw"].update(hw_update=True)

    def random_transformation(self):
        # choose to do a partitioning transform of change a variable
        transform = np.random.choice(["partition", "variable"], p=[0.1,0.9])
        # pick a random partition
        partition = random.choice(self.network.partitions)
        # get partition index
        partition_index = self.network.partitions.index(partition)

        # pick a random transform
        if transform == "partition":
            transform_type = random.choice(["split", "merge"])
            if transform_type == "split":
                # choose random split nodes
                valid_splits = self.network.valid_splits(partition_index)
                if valid_splits:
                    nodes = random.choice(valid_splits)
                    # apply split
                    self.network.split(partition_index, nodes)
            elif transform_type == "merge":
                # choose random merge
                valid_merges = self.network.valid_merges()
                if valid_merges:
                    merge = random.choice(valid_merges)
                    self.network.merge(merge)
        else:
            # pick a random layer
            layer = random.choice(list(partition.nodes()))
            node_hw = partition.nodes[layer]["hw"]
            # pick a random variable
            variable = random.choices(self.valid_variables)[0]
            # apply a random value to that variable (within constraints)
            if variable == "channel_in_folding":
                folding = random.choices(node_hw.valid_channel_in_folding)[0]
                node_hw.channel_in_folding = folding
                partition.folding_match(layer, folding, "io")
            elif variable == "channel_out_folding":
                folding = random.choices(node_hw.valid_channel_out_folding)[0]
                node_hw.channel_out_folding = folding
                partition.folding_match(layer, folding, "io")
            elif variable == "kernel_folding":
                node_hw.kernel_folding = random.choices(node_hw.valid_kernel_folding)[0]

    def optimise(self):

        log = []
        graph_index = 0

        # keep iterating until we meet the minimum temperature
        while self.T_min < self.T:

            # get the throughput of the current network state
            cost = self.network.eval_cost()

            # keep a copy of the current network state
            network_copy = copy.deepcopy(self.network)

            # perform a number of permutations of this network
            for _ in range(self.iterations):
                self.random_transformation()

            # update the network
            self.update()

            # check the network is within platform resource constraints
            if not self.network.check_constraints():
                self.network = network_copy
                continue

            # log the current resources and cost
            new_cost = self.network.eval_cost()
            chosen = True

            # perform the annealing descision
            if math.exp(min(0,(cost - new_cost)/(self.k*self.T))) < random.uniform(0,1):
                self.network = network_copy
                chosen = False

            # reduce temperature
            self.T *= self.cool

            # update the log
            if chosen:
                self.network.update_fpgaconvnet()
                # save log of resources and performance
                with open(f"outputs/log/{graph_index}.json", "w") as f:
                    json.dump({
                        "latency": self.network.eval_latency(),
                        "throughput": self.network.eval_throughput(),
                        "resource" : self.network.eval_resource()
                    }, f)
                # export to fpgaconvnet and save visualisation
                self.network.fpgaconvnet_net.visualise(f"outputs/vis/{graph_index}.dot")

                # increment graph index
                graph_index += 1
