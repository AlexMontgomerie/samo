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

    @property
    def channels_out(self):
        return self.channels

    def latency_in(self, eval=False):
        if eval:
            return self.rows*self.cols*self.channels/self.streams_in.value[0]
        else:
            return self.rows*self.cols*self.channels/self.streams_in

    def latency_out(self, eval=False):
        if eval:
            return self.rows*self.cols*self.channels/self.streams_out.value[0]
        else:
            return self.rows*self.cols*self.channels/self.streams_out

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : 0,
             "BRAM" : 0,
             "FF" : 0
        }


