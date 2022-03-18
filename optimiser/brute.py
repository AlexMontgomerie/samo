import itertools
import copy
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .network import Network

@dataclass
class BruteForce:
    network: Network

    def update(self, config):
        assert len(self.network.partitions) == 1

        layers = []
        for index, layer in enumerate(self.network.partitions[0]):
            self.network.partitions[0].nodes[layer]["hw"].channel_in_folding    = config[index][0]
            self.network.partitions[0].nodes[layer]["hw"].channel_out_folding   = config[index][1]
            self.network.partitions[0].nodes[layer]["hw"].kernel_folding        = config[index][2]
            self.network.partitions[0].nodes[layer]["hw"].update(hw_update=True)
            layers.append(layer)


        index = 0
        while index < len(config):
            if config[index][3]:
                for partition_index, partition in enumerate(self.network.partitions):
                    if layers[index] in partition:
                        for next_node in partition.successors(layers[index]):
                            self.network.split(partition_index, (layers[index], next_node))
            index += 1
        

    def optimise(self):
        assert len(self.network.partitions) == 1

        # get all the configurations
        size = 1
        configurations = []
        for layer in self.network.partitions[0]:

            layer_split = [False]
            for split_pair in self.network.valid_splits(0):
                if split_pair[0] == layer:
                    layer_split = [False, True]
                    break

            layer_configurations = list(itertools.product(
                self.network.partitions[0].nodes[layer]["hw"].valid_channel_in_folding,
                self.network.partitions[0].nodes[layer]["hw"].valid_channel_out_folding,
                self.network.partitions[0].nodes[layer]["hw"].valid_kernel_folding,
                layer_split
                ))

            configurations.append(layer_configurations)
            size *= len(layer_configurations)
        configurations = itertools.product(*configurations)
        print(f"full configuration space size : {size}")

        network_init = copy.deepcopy(self.network)

        # track all valid networks
        valid_configs = {}

        # iterate over all the configurations
        for config in tqdm(configurations, desc="evaluating configurations"):
            # update the network
            self.network = copy.deepcopy(network_init)
            self.update(config)
            # evaluate the cost
            cost  = self.network.eval_cost()
            # if network is within constraints, log the network and it's cost
            if self.network.check_constraints():
                valid_configs[config] = cost

        # find the network with the lowest cost
        best_config = min(valid_configs, key=valid_configs.get_cost)

        # update with the best configuration
        self.network = copy.deepcopy(network_init)
        self.update(best_config)
