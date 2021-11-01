import uuid
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
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))

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

    def latency_in(self):
        return 1

    def latency_out(self, eval=False):
        return 1

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : 0,
             "BRAM" : 0,
             "FF" : 0
        }


