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
        self.constraints = { "matching_folding" : type(layer) not in [Dense, Conv2D] }

    def latency(self):
        return max(
                np.prod(self.layer.get_input_variable().shape) // self.channel_in_folding,
                np.prod(self.layer.get_output_variable().shape) // self.channel_out_folding )

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : self.channel_in_folding*self.channel_out_folding,
             "BRAM" : 0,
             "FF" : 0
        }
