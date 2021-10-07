import math
from dataclasses import dataclass

from gekko import GEKKO
from gekko.gk_variable import GKVariable

@dataclass
class ReLU:
    rows: int
    cols: int
    channels: int
    streams_in: GKVariable = None
    streams_out: GKVariable = None

    def latency_in(self):
        return self.rows*self.cols*self.channels/self.streams_in

    def latency_out(self):
        return self.rows*self.cols*self.channels/self.streams_out

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : 0,
             "BRAM" : 0,
             "FF" : 0
        }


