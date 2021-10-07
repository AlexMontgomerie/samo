import math
from dataclasses import dataclass

from gekko import GEKKO
from gekko.gk_variable import GKVariable

@dataclass
class InnerProduct:
    channels: int
    filters: int
    streams_in: GKVariable = None
    streams_out: GKVariable = None

    def latency_in(self):
        return self.channels/self.streams_in

    def latency_out(self):
        return self.filters/self.streams_out

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : self.streams_in*self.streams_out,
             "BRAM" : 0,
             "FF" : 0
        }
