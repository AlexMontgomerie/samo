from optimiser import Network

class FinnNetworkWrapper(Network):

    def validate(self):
        # iterate over nodes in the network
        for node in self.nodes:
            # if the layer is fully connected layer, make sure only the previous folding matches
            if self.nodes[node]["hw"].finn_node.onnx_node.op_type == "StreamingFCLayer_Batch":
                # get the previous node
                prev_node = list(self.predecessors(node))[0]
                if self.nodes[prev_node]["hw"].finn_node.onnx_node.op_type != "StreamingFCLayer_Batch":
                    assert self.nodes[prev_node]["hw"].channel_out_folding == self.nodes[node]["hw"].channel_in_folding
        return True

    def check_constraints(self):
        return Network.check_constraints(self) and self.validate()
