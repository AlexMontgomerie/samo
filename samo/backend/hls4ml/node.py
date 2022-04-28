import math
import numpy as np

from hls4ml.model.hls_layers import Layer, Dense, Conv2D, Conv1D
from samo.model import Node

def _check_conditions(n_in, n_out, rf):
    # from https://github.com/fastmachinelearning/hls4ml/blob/7f75add50a5acd2a4335bde0ab98c9d4e79e1137/hls4ml/templates/vivado_template.py#L448
    multfactor = min(n_in, rf)
    multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
    _assert = (((multiplier_limit % n_out) == 0) or (rf >= n_in))
    _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
    _assert = _assert and (((n_in * n_out) % rf) == 0)
    return _assert

class HLS4MLNodeWrapper(Node):

    def __init__(self, layer: Layer):

        # save layer type settings
        self.layer_type = type(layer)

        # get the size of input and output featuremaps
        self.size_in    = np.prod(layer.get_input_variable().shape)
        self.size_out   = np.prod(layer.get_output_variable().shape)

        # get the channel dimensions
        self.channels_in    = layer.get_input_variable().shape[-1]
        self.channels_out   = layer.get_output_variable().shape[-1]

        # get the spatial dimensions
        if len(layer.get_input_variable().shape) == 3:
            self.spatial_size = np.prod(layer.get_input_variable().shape[:-1])
        else:
            self.spatial_size = 1

        # get the kernel size
        if self.layer_type in [Conv1D, Conv2D]:
            self.kernel_size = layer.get_attr("filt_width")

        # set the matching folding constraint
        self.constraints = { "matching_intra_folding" : False,
                             "matching_inter_folding" : False,
                             "divisible_inter_folding" : False}

        # get all valid kernel folding
        self._valid_kernel_folding = []
        for rf in range(1, self.channels_in*self.channels_out*self.kernel_size*self.kernel_size+1):
            if _check_conditions(self.channels_in*self.kernel_size*self.kernel_size, self.channels_out, rf):
                self._valid_kernel_folding.append(rf)

        # start with the smallest design
        #if self.layer_type in [Conv1D, Conv2D, Dense]:
        #    self.kernel_folding = self._valid_kernel_folding[-1]
        self.kernel_folding = 1

    def get_reuse_factor(self):
        # return self.kernel_folding
        return int(self.channels_in*self.channels_out*\
                self.kernel_size*self.kernel_size/self.kernel_folding)

    @property
    def valid_channel_in_folding(self):
        return [1]

    @property
    def valid_channel_out_folding(self):
        return [1]

    @property
    def valid_kernel_folding(self):
        return self._valid_kernel_folding

    def latency(self):
        return self.spatial_size*self.get_reuse_factor()

    def resource(self):
        if self.layer_type in [Conv1D, Conv2D, Dense]:
             #dsp_usage = int(self.channels_in*self.channels_out*self.kernel_size*self.kernel_size/self.get_reuse_factor())
             dsp_usage = self.kernel_folding
        else:
            dsp_usage = 0
        return {
             "LUT" : 0,
             "DSP" : dsp_usage,
             "BRAM" : 0,
             "FF" : 0
        }
