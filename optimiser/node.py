import math
from functools import reduce
from dataclasses import dataclass, field

def get_factors(n):
    return list(set(reduce(list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

@dataclass
class Node:
    channels_in: int
    channels_out: int
    kernel_size: int = 1
    channel_in_folding: int = 1
    channel_out_folding: int = 1
    kernel_folding: int = 1
    constraints: dict = field(default_factory=lambda: {"matching_intra_folding": True, "matching_inter_folding": False, "divisible_inter_folding": True})

    def __hash__(self):
        return hash(repr(self))

    @property
    def valid_channel_in_folding(self):
        return get_factors(self.channels_in)

    @property
    def valid_channel_out_folding(self):
        return get_factors(self.channels_out)

    @property
    def valid_kernel_folding(self):
        return get_factors(self.kernel_size*self.kernel_size)

    def check_matching_intra_folding(self):
        assert self.channel_in_folding == self.channel_out_folding
        return True

    def check_constraints(self):
        # check all the constraints
        constraints = []
        constraints += [self.check_matching_intra_folding if self.constraints["matching_intra_folding"] else True]
        constraints += [self.channel_in_folding in self.valid_channel_in_folding]
        constraints += [self.channel_out_folding in self.valid_channel_out_folding]
        constraints += [self.kernel_folding in self.valid_kernel_folding]
        # ensure all constraints are held
        return reduce(lambda a, b: a and b, constraints)

    def update(self):
        pass

    def latency(self):
        return 1

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : 0,
             "BRAM" : 0,
             "FF" : 0
        }


