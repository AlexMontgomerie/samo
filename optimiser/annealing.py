import csv
import math
import copy
import random
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from .network import Network

@dataclass
class SimulatedAnnealing:
    network: Network
    T: float = 10.0
    k: float = 100.0
    T_min: float = 0.01
    cool: float = 0.98
    iterations: int = 10
    valid_variables: list = field(default_factory=lambda: ["channel_in_folding", "channel_out_folding", "kernel_folding"])

    def update(self):
        for partition in self.network.partitions:
            for index, layer in enumerate(partition):
                partition.nodes[layer]["hw"].update(hw_update=True)

    def random_transformation(self):
        # pick a random partition
        partition = random.choice(self.network.partitions)
        transform = np.random.choice(["partition", "variable"], p=[0.05,0.95])
        # pick a random transform
        if transform == "partition":
            transform_type = random.choice(["split", "merge"])
            if transform_type == "split":
                # get partition index
                partition_index = self.network.partitions.index(partition)
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

        def generator():
            while self.T_min < self.T:
                yield

        log = []

        # keep iterating until we meet the minimum temperature
        pbar = tqdm(generator())
        for _ in pbar:

            # update the description
            pbar.set_description(desc=f"simulated annealing iterations (T={self.T:.3f})")

            # get the throughput of the current network state
            latency = self.network.eval_latency()

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

            # log the current resources and latency
            new_latency = self.network.eval_latency()
            # new_resource = self.network.eval_resource()
            chosen = True

            # perform the annealing descision
            # print(math.exp(min(0,(latency - self.network.eval_latency())/(self.k*self.T))), latency-self.network.eval_latency(), self.T)
            if math.exp(min(0,(latency - new_latency)/(self.k*self.T))) < random.uniform(0,1):
                self.network = network_copy
                chosen = False

            # reduce temperature
            self.T *= self.cool

            # # update the log
            # log += [[
            #         new_latency,
            #         new_resource["BRAM"],
            #         new_resource["DSP"],
            #         new_resource["LUT"],
            #         new_resource["FF"],
            #         chosen
            # ]]

        # write log to a file
        with open("outputs/log.csv", "w") as f:
            writer = csv.writer(f)
            [ writer.writerow(row) for row in log ]
