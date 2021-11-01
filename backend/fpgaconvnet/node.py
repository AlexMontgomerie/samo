from fpgaconvnet_optimiser.models.layers import Layer, Convolution
from optimiser import Node

class FPGAConvNetWrapper(Node):

    def __init__(self, layer: Layer):
        self.layer = layer

        self.channels_in    = layer.channels_in
        self.channels_out   = layer.channels_out

        if type(layer) == Convolution:
            self.kernel_size = layer.kernel_size

    def update(self):
        self.layer.coarse_in = self.channels_in // self.channels_in_folding
        self.layer.coarse_out = self.channels_out // self.channels_out_folding
        self.layer.fine = (self.kernel_size*self.kernel_size) // self.kernel_folding

    def latency_in(self):
        return self.layer.latency_in()

    def latency_out(self):
        return self.layer.latency_out()

    def resource(self):
        return self.layer.resource()
