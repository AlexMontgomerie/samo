from networkx import DiGraph

class FinnNetworkWrapper(DiGraph):
    def __init__(self):
        super().__init__()

    def validate(self):
        for i, n in enumerate(self.nodes):
            if n.finn_node.onnx_node.op_type == "StreamingFCLayer_Batch":
                prev = list(self.nodes)[i-1]
                if prev.finn_node.onnx_node.op_type != "StreamingFCLayer_Batch":
                    if prev.channel_out_folding != n.channel_in_folding:
                        return False
            else:
                if n.channel_in_folding != n.channel_out_folding:
                    return False

        return True