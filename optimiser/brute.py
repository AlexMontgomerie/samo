import itertools
import copy
from dataclasses import dataclass
import numpy as np

from .optimiser import Optimiser

@dataclass
class BruteForce(Optimiser):

    def update(self, config):
        for index, layer in enumerate(self.network):
            layer.channel_in_folding    = config[index][0]
            layer.channel_out_folding   = config[index][1]
            layer.kernel_folding        = config[index][2]
            layer.update()

    def optimise(self):

        # get all the configurations
        configurations = []
        for layer in self.network:
            configurations.append(list(itertools.product(
                layer.valid_channel_in_folding,
                layer.valid_channel_out_folding,
                layer.valid_kernel_folding)))
        configurations = list(itertools.product(*configurations))

        # track all valid networks
        valid_configs = {}

        # iterate over all the configurations
        for config in configurations:
            # update the network
            self.update(config)
            # evaluate the latency
            latency  = self.eval_latency()
            # if network is within constraints, log the network and it's latency
            if self.check_resource_constraints():
                valid_configs[config] = latency

        # find the network with the lowest latency
        best_config = min(valid_configs, key=valid_configs.get)

        # update with the best configuration
        self.update(best_config)

