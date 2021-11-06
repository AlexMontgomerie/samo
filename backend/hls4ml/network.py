from optimiser import Network

class HLS4MLNetworkWrapper(Network):

    def check_inter_layer_matching_folding(self):
        # iterate over the nodes in the network
        for node in self.nodes:
            # iterate over all the nodes before
            for prev_node in self.predecessors(node):
                if self.nodes[prev_node]["hw"].channel_out_folding != self.nodes[node]["hw"].channel_in_folding:
                    return False
            # iterate over all the nodes after
            for next_node in self.successors(node):
                if self.nodes[node]["hw"].channel_out_folding != self.nodes[next_node]["hw"].channel_in_folding:
                    return False
        # if there's no clashes, then return true
        return True

    def check_constraints(self):
        return Network.check_constraints(self) and self.check_inter_layer_matching_folding()
