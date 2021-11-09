import numpy as np

from hls4ml.model.hls_layers import Layer, Dense, Conv2D
from optimiser import Node

class HLS4MLNodeWrapper(Node):

    def __init__(self, layer: Layer):
        self.layer = layer

        # get the channel dimensions
        self.channels_in    = layer.get_input_variable().shape[-1]
        self.channels_out   = layer.get_output_variable().shape[-1]

        # set the matching folding constraint
        self.constraints = { "matching_folding" : False }

    def get_reuse_factor(self):
        return int((self.channels_in*self.channels_out)/(self.channel_in_folding*self.channel_out_folding))

    @property
    def valid_channel_in_folding(self):
        return list(np.arange(self.channels_in)+1)

    @property
    def valid_channel_out_folding(self):
        return list(np.arange(self.channels_out)+1)

    def latency(self):
        return self.get_reuse_factor() if type(self.layer) in [Dense, Conv2D] else 1

    def resource(self):
        dsp = int(self.channels_in*self.channels_out/self.get_reuse_factor()) if type(self.layer) in [Dense, Conv2D] else 0
        return {
             "LUT" : 0,
             "DSP" : dsp,
             "BRAM" : 0,
             "FF" : 0
        }
