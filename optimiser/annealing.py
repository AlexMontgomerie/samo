import math
import copy
import random
from dataclasses import dataclass, field
import numpy as np

from .optimiser import Optimiser

@dataclass
class SimulatedAnnealing(Optimiser):
    T: float = 10.0
    k: float = 100.0
    T_min: float = 0.001
    cool: float = 0.98
    iterations: int = 100
    valid_variables: list = field(default_factory=lambda: ["channel_in_folding", "channel_out_folding", "kernel_folding"])

    def update(self):
        for index, layer in enumerate(self.network):
            layer.update()

    def random_transformation(self):
        # pick a random layer
        layer = random.choices(list(self.network.nodes()))[0]
        # pick a random variable
        variable = random.choices(self.valid_variables)[0]
        # apply a random value to that variable (within constraints)
        if variable == "channel_in_folding":
            layer.channel_in_folding = random.choices(layer.valid_channel_in_folding)[0]
        elif variable == "channel_out_folding":
            layer.channel_out_folding = random.choices(layer.valid_channel_out_folding)[0]
        elif variable == "kernel_folding":
            layer.kernel_folding = random.choices(layer.valid_kernel_folding)[0]

    def optimise(self):

        # keep iterating until we meet the minimum temperature
        while self.T_min < self.T:

            # get the throughput of the current network state
            latency = self.eval_latency()

            # keep a copy of the current network state
            network_copy = copy.deepcopy(self.network)

            # perform a number of permutations of this network
            for _ in range(self.iterations):
                self.random_transformation()

            # update the network
            self.update()

            # perform the annealing descision
            if math.exp(min(0,(latency - self.eval_latency())/(self.k*self.T))) < random.uniform(0,1):
                self.network = network_copy

            # check the network is within platform resource constraints
            if not self.check_resource_constraints():
                self.network = network_copy

            # reduce temperature
            self.T *= self.cool
