from fpgaconvnet.models.layers import Layer, ConvolutionLayerBase, InnerProductLayer
from samo.model import Node

class FPGAConvNetWrapper(Node):

    def __init__(self, layer: Layer, batch_size=1):

        # store the batch size
        self.batch_size = batch_size

        # store the fpgaconvnet layer
        self.layer = layer

        # get the size in and out
        self.size_in = layer.rows_in()*layer.cols_in()*layer.channels_in()
        self.size_out = layer.rows_out()*layer.cols_out()*layer.channels_out()

        # get the channels in and out for the layer
        self.channels_in    = layer.channels_in()
        self.channels_out   = layer.channels_out()

        # add the kernel size if it's a convolution layer
        if issubclass(type(layer), ConvolutionLayerBase):
            self.kernel_size = layer.kernel_size[0]

        # set the matching folding constraint
        self.constraints = { "matching_intra_folding" : not issubclass(type(layer),
            ConvolutionLayerBase) and not isinstance(layer, InnerProductLayer),
                             "matching_inter_folding" : True,
                             "divisible_inter_folding" : True}

    def update(self, hw_update=False):
        self.layer.coarse_in = self.channel_in_folding
        self.layer.coarse_out = self.channel_out_folding
        if hasattr(self.layer, "fine"):
            self.layer.fine = self.kernel_folding
        if hw_update:
            self.layer.update()

    def latency(self):
        return self.layer.latency()*self.batch_size

    def resource(self):
        return self.layer.resource()

