import numpy as np

from hls4ml.model.hls_layers import Layer
from optimiser import Node

class HLS4MLWrapper(Node):

    def __init__(self, layer: Layer):
        self.layer = layer

        self.channels_in    = layer.get_input_variable().dim_names[-1]
        self.channels_out   = layer.get_output_variable().dim_names[-1]

    def latency_in(self):
        return np.prod(layer.get_input_variable().dim_names) // self.channel_in_folding

    def latency_out(self):
        return np.prod(layer.get_output_variable().dim_names) // self.channel_out_folding

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : self.channel_in_folding*self.channel_out_folding,
             "BRAM" : 0,
             "FF" : 0
        }
