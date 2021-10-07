from dataclasses import dataclass

from gekko import GEKKO
from gekko.gk_variable import GKVariable

@dataclass
class Layer:
    rows: int
    cols: int
    channels: int
    streams_in: GKVariable
    streams_out: GKVariable

    def latency(self):
        return self.rows*self.cols*self.channels/self.streams_in

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : self.streams_in*self.streams_out,
             "BRAM" : 0,
             "FF" : 0
        }


