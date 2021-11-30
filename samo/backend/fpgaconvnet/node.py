from fpgaconvnet_optimiser.models.layers import Layer, ConvolutionLayer, InnerProductLayer
from samo.model import Node

class FPGAConvNetWrapper(Node):

    def __init__(self, layer: Layer):

        # store the fpgaconvnet layer
        self.layer = layer

        # get the channels in and out for the layer
        self.channels_in    = layer.channels_in()
        self.channels_out   = layer.channels_out()

        # add the kernel size if it's a convolution layer
        if type(layer) == ConvolutionLayer:
            self.kernel_size = layer.kernel_size[0]

        # set the matching folding constraint
        self.constraints = { "matching_intra_folding" : type(layer) not in [ConvolutionLayer, InnerProductLayer],
                             "matching_inter_folding" : False,
                             "divisible_inter_folding" : False}

    def update(self):
        self.layer.coarse_in = self.channel_in_folding
        self.layer.coarse_out = self.channel_out_folding
        if hasattr(self.layer, "fine"):
            self.layer.fine = self.kernel_folding

    def latency(self):
        return self.layer.latency()

    def resource(self):
        return self.layer.resource()
