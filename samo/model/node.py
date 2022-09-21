import logging
import math
from functools import reduce
from dataclasses import dataclass, field
from typing import List

def get_factors(n):
    """
    helper function for returning all the factors of `n`
    """
    return list(set(reduce(list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

@dataclass
class Node:
    """
    A generalisation of hardware nodes in Streaming Architectures. This class
    represents the Node described in the paper. Hardware in the backend's
    parsed network representation is mapped to this class.
    """

    size_in: int
    """
    number of words from incoming feature-map into the node
    """
    size_out: int
    """
    number of words from outgoing feature-map from the node
    """
    channels_in: int
    """
    channel dimension of the incoming feature-map
    """
    channels_out: int
    """
    channel dimension of the outgoing feature-map
    """
    kernel_size: int = 1
    """
    kernel dimension of the node. Used to determine the valid kernel folding
    factors. The default is 1.
    """
    channel_in_folding: int = 1
    """
    channel in folding variable. Defines the level of parallelism across the
    incoming feature-map's channel dimension.
    The default is 1.
    """
    channel_out_folding: int = 1
    """
    channel out folding variable. Defines the level of parallelism across the
    outgoing feature-map's channel dimension.
    The default is 1.
    """
    kernel_folding: int = 1
    """
    kernel folding variable. This is parallelism across internal operations,
    defined by kernel size and valid kernel folding.
    The default is 1.
    """
    constraints: dict = field(default_factory=lambda: {"matching_intra_folding": True, "matching_inter_folding": False, "divisible_inter_folding": True})
    """
    dictionary of constraints on the node. Each key is a description of the
    constraint, and the value is a Boolean of whether it is to be applied or
    not.
    The defaults are:
    ```
    {
        "matching_intra_folding": True,
        "matching_inter_folding": False,
        "divisible_inter_folding": True
    }
    ```
    """

    def __hash__(self):
        return hash(repr(self))

    @property
    def valid_channel_in_folding(self) -> List[int]:
        return sorted(get_factors(self.channels_in))

    @property
    def valid_channel_out_folding(self):
        return sorted(get_factors(self.channels_out))

    @property
    def valid_kernel_folding(self):
        return sorted(get_factors(self.kernel_size*self.kernel_size))

    def check_matching_intra_folding(self):
        intra_folding_matching = self.channel_in_folding == self.channel_out_folding
        if not intra_folding_matching:
            logging.error(f"{self.channel_in_folding} != {self.channel_out_folding}")
        return intra_folding_matching

    def check_constraints(self):
        # check all the constraints
        constraints = []

        # intra folding matching
        if self.constraints["matching_intra_folding"]:
            logging.info("checking matching intra folding")
            constraints += [self.check_matching_intra_folding]

        # channel in folding
        logging.info("checking input channel folding valid")
        constraints += [self.channel_in_folding in self.valid_channel_in_folding]

        # channel out folding
        logging.info("checking output channel folding valid")
        constraints += [self.channel_out_folding in self.valid_channel_out_folding]

        # kernel folding
        logging.info("checking kernel folding valid")
        constraints += [self.kernel_folding in self.valid_kernel_folding]

        # ensure all constraints are held
        return reduce(lambda a, b: a and b, constraints)

    def update(self, hw_update=False):
        pass # to be implemented by backend

    def latency(self):
        return 1

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : 0,
             "BRAM" : 0,
             "FF" : 0
        }

    def reset(self):
        self.channel_in_folding = 1
        self.channel_out_folding = 1
        self.kernel_folding = 1

        self.update(hw_update=True)

