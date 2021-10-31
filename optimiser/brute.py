import itertools
import copy
from dataclasses import dataclass
import numpy as np

from .optimiser import Optimiser

@dataclass
class BruteForce(Optimiser):

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
        valid_networks = {}

        # iterate over all the configurations
        for config in configurations:
            # update the network
            for index, layer in enumerate(self.network):
                layer.channel_in_folding    = config[index][0]
                layer.channel_out_folding   = config[index][1]
                layer.kernel_folding        = config[index][2]
            # evaluate the latency
            latency  = self.eval_latency()
            resource = self.eval_resource()
            # check constraints
            within_bram = resource["BRAM"] <= self.platform["BRAM"]
            within_ff   = resource["FF"]   <= self.platform["FF"]
            within_lut  = resource["LUT"]  <= self.platform["LUT"]
            within_dsp  = resource["DSP"]  <= self.platform["DSP"]
            # if network is within constraints, log the network and it's latency
            if within_bram and within_ff and within_lut and within_dsp:
                valid_networks[copy.copy(self.network)] = latency

        # find the network with the lowest latency
        best_network = min(valid_networks, key=valid_networks.get)

        # return the best network
        return best_network

