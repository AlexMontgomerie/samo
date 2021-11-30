import itertools
import copy
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from samo.model.network import Network

@dataclass
class BruteForce:
    network: Network

    def update(self, config):
        for index, layer in enumerate(self.network):
            self.network.nodes[layer]["hw"].channel_in_folding    = config[index][0]
            self.network.nodes[layer]["hw"].channel_out_folding   = config[index][1]
            self.network.nodes[layer]["hw"].kernel_folding        = config[index][2]
            self.network.nodes[layer]["hw"].update()

    def optimise(self):

        # get all the configurations
        size = 1
        configurations = []
        for layer in tqdm(self.network, desc="collecting configurations"):
            layer_configurations = list(itertools.product(
                self.network.nodes[layer]["hw"].valid_channel_in_folding,
                self.network.nodes[layer]["hw"].valid_channel_out_folding,
                self.network.nodes[layer]["hw"].valid_kernel_folding))
            configurations.append(layer_configurations)
            size *= len(layer_configurations)
        configurations = itertools.product(*configurations)

        print(f"full configuration space size : {size}")

        # get all folding-matched configurations
        configurations = []
        for layer in tqdm(self.network, desc="collecting intra folding matching configurations"):

            layer_configurations = list(itertools.product(
                self.network.nodes[layer]["hw"].valid_channel_in_folding,
                self.network.nodes[layer]["hw"].valid_channel_out_folding,
                self.network.nodes[layer]["hw"].valid_kernel_folding))

            if self.network.nodes[layer]["hw"].constraints["matching_intra_folding"]:
                layer_configurations = list(filter(lambda x: x[0] == x[1], layer_configurations))
            configurations.append(layer_configurations)

        configurations = itertools.product(*configurations)
        for i, layer in enumerate(tqdm(self.network, desc="collecting inter folding matching configurations")):
            if self.network.nodes[layer]["hw"].constraints["matching_inter_folding"] and self.network.out_degree(layer) > 0:
                configurations = filter(lambda x,i=i: x[i][1] == x[i+1][0], configurations)
            if self.network.nodes[layer]["hw"].constraints["divisible_inter_folding"] and self.network.out_degree(layer) > 0:
                configurations = filter(lambda x,i=i: max(x[i][1], x[i+1][0]) % min(x[i][1], x[i+1][0]) == 0, configurations)

        size = 0
        for _ in tqdm(copy.deepcopy(configurations), desc="counting space size"):
            size += 1

        print(f"folding-matched configuration space size : {size}")

        # track all valid networks
        valid_configs = {}

        # iterate over all the configurations
        for config in tqdm(configurations, desc="evaluating configurations"):
            # update the network
            self.update(config)
            # evaluate the latency
            latency  = self.network.eval_latency()
            # if network is within constraints, log the network and it's latency
            if self.network.check_constraints():
                valid_configs[config] = latency

        # find the network with the lowest latency
        best_config = min(valid_configs, key=valid_configs.get)

        # update with the best configuration
        self.update(best_config)

