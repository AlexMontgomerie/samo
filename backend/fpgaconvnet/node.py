import uuid

from fpgaconvnet_optimiser.models.layers import Layer, ConvolutionLayer
from optimiser import Node

class FPGAConvNetWrapper(Node):

    def __init__(self, layer: Layer):

        self.node_id = str(uuid.uuid4()) # hack to get it to recognise the node as unique

        # store the fpgaconvnet layer
        self.layer = layer

        # get the channels in and out for the layer
        self.channels_in    = layer.channels_in()
        self.channels_out   = layer.channels_out()

        # add the kernel size if it's a convolution layer
        if type(layer) == ConvolutionLayer:
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
