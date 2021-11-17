from finn.util.basic import get_by_name

from optimiser import Node

class FinnNodeWrapper(Node):
    def __init__(self, finn_node, channels_in, channels_out):

        self.finn_node = finn_node

        # get the channel dimensions
        self.channels_in = channels_in
        self.channels_out = channels_out

        # set the matching folding constraint
        self.constraints = { "matching_intra_folding" : finn_node.onnx_node.op_type not in ["StreamingFCLayer_Batch"],
                             "matching_inter_folding": finn_node.onnx_node.op_type not in ["StreamingFCLayer_Batch"]}

    def update(self):
        if get_by_name(self.finn_node.onnx_node.attribute, "SIMD") is not None:
            self.finn_node.set_nodeattr("SIMD", self.channel_in_folding)

        if get_by_name(self.finn_node.onnx_node.attribute, "PE") is not None:
            self.finn_node.set_nodeattr("PE", self.channel_out_folding)

    def latency(self):
        return self.finn_node.get_exp_cycles()

    def resource(self):
        return {
             "LUT" : self.finn_node.lut_estimation(),
             "DSP" : self.finn_node.dsp_estimation(),
             "BRAM" : self.finn_node.bram_estimation(),
             "FF" : 0
        }
