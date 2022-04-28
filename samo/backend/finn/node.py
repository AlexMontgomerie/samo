import numpy as np

from finn.util.basic import get_by_name

from samo.model import Node

class FinnNodeWrapper(Node):
    def __init__(self, finn_node, size_in, size_out):

        self.finn_node = finn_node

        # get the channel dimensions
        self.channels_in = size_in[-1]
        self.channels_out = size_out[-1]

        self.size_in = np.prod(size_in)
        self.size_out = np.prod(size_out)

        # set the matching folding constraint
        self.constraints = { "matching_intra_folding" : finn_node.onnx_node.op_type not in ["StreamingFCLayer_Batch"],
                             "matching_inter_folding": finn_node.onnx_node.op_type not in ["StreamingFCLayer_Batch", "DuplicateStreams_Batch"],
                             "divisible_inter_folding": finn_node.onnx_node.op_type in ["StreamingFCLayer_Batch", "DuplicateStreams_Batch"],}

        self.split = finn_node.onnx_node.op_type in ["StreamingFCLayer_Batch", "Vector_Vector_Activate_Batch"]

    def update(self, hw_update=False):
        if hw_update:
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
