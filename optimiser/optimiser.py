from dataclasses import dataclass, field
import networkx as nx

@dataclass
class Optimiser:
    network: nx.DiGraph
    platform: dict
    constraints: dict = field(default_factory=lambda: {"resource" : True, "inter_layer_matching" : True})

    def eval_latency(self):
        #max_latency_in  = max([ self.network.nodes[layer]["hw"].latency_in() for layer in self.network])
        #max_latency_out = max([ self.network.nodes[layer]["hw"].latency_out() for layer in self.network])
        max_latency_in  = max([ layer.latency_in() for layer in self.network])
        max_latency_out = max([ layer.latency_out() for layer in self.network])
        return max(max_latency_in, max_latency_out)

    def eval_resource(self):
        #return {
        #    "LUT"   : sum([ self.network.nodes[layer]["hw"].resource()["LUT"]  for layer in self.network]),
        #    "DSP"   : sum([ self.network.nodes[layer]["hw"].resource()["DSP"]  for layer in self.network]),
        #    "BRAM"  : sum([ self.network.nodes[layer]["hw"].resource()["BRAM"] for layer in self.network]),
        #    "FF"    : sum([ self.network.nodes[layer]["hw"].resource()["FF"]   for layer in self.network])
        #}
       return {
            "LUT"   : sum([ layer.resource()["LUT"]  for layer in self.network]),
            "DSP"   : sum([ layer.resource()["DSP"]  for layer in self.network]),
            "BRAM"  : sum([ layer.resource()["BRAM"] for layer in self.network]),
            "FF"    : sum([ layer.resource()["FF"]   for layer in self.network])
        }

    def check_resource_constraints(self):
        # get the current resource usage
        resource = self.eval_resource()
        # check within the platform constraints
        within_bram = resource["BRAM"] <= self.platform["BRAM"]
        within_ff   = resource["FF"]   <= self.platform["FF"]
        within_lut  = resource["LUT"]  <= self.platform["LUT"]
        within_dsp  = resource["DSP"]  <= self.platform["DSP"]
        # if network is within constraints, return true
        if within_bram and within_ff and within_lut and within_dsp:
            return True
        return False

    def check_inter_layer_matching_folding(self):
        # iterate over the nodes in the network
        for node in self.network:
            # iterate over all the nodes before
            for prev_node in self.network.predecessors(node):
                if self.network.nodes[prev_node]["hw"].channel_out_folding != self.network.nodes[node]["hw"].channel_in_folding:
                    return False
            # iterate over all the nodes after
            for next_node in self.network.successors(node):
                if self.network.nodes[node]["hw"].channel_out_folding != self.network.nodes[next_node]["hw"].channel_in_folding:
                    return False
        # if there's no clashes, then return true
        return True

    def check_constraints(self):
        # check all the constraints (if they are required)
        #resources = self.check_resource_constraints() if self.constraints["resource"] else True
        #inter_layer_matching = self.check_inter_layer_matching_folding() if self.constraints["inter_layer_matching"] else True
        # ensure it's within all constraints
        #if resources and inter_layer_matching:
        #    return True
        #return False

        return self.check_resource_constraints() and self.network.validate()

    def optimise(self):
        pass

def load_from_opt_network(network, opt_network):
    for i, opt_node in enumerate(opt_network.nodes()):
        node = list(network.nodes())[i]
        node.channel_in_folding = opt_node.channel_in_folding
        node.channel_out_folding = opt_node.channel_out_folding
        node.kernel_folding = opt_node.kernel_folding
        node.update()
