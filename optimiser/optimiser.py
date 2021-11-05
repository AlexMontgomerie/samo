from dataclasses import dataclass, field
from functools import reduce
import networkx as nx

@dataclass
class Optimiser:
    network: nx.DiGraph
    platform: dict
    constraints: dict = field(default_factory=lambda: {"resource" : True, "inter_layer_matching" : True})

    def eval_latency(self):
        max_latency_in  = max([ self.network.nodes[layer]["hw"].latency_in() for layer in self.network])
        max_latency_out = max([ self.network.nodes[layer]["hw"].latency_out() for layer in self.network])
        return max(max_latency_in, max_latency_out)

    def eval_resource(self):
        return {
            "LUT"   : sum([ self.network.nodes[layer]["hw"].resource()["LUT"]  for layer in self.network]),
            "DSP"   : sum([ self.network.nodes[layer]["hw"].resource()["DSP"]  for layer in self.network]),
            "BRAM"  : sum([ self.network.nodes[layer]["hw"].resource()["BRAM"] for layer in self.network]),
            "FF"    : sum([ self.network.nodes[layer]["hw"].resource()["FF"]   for layer in self.network])
        }

    def check_resource_constraints(self):
        # get the current resource usage
        resource = self.eval_resource()
        # check within the platform constraints
        rsc_constraints = []
        rsc_constraints += [resource["BRAM"] <= self.platform["BRAM"]]
        rsc_constraints += [resource["FF"]   <= self.platform["FF"]]
        rsc_constraints += [resource["LUT"]  <= self.platform["LUT"]]
        rsc_constraints += [resource["DSP"]  <= self.platform["DSP"]]
        # if network is within constraints, return true
        return reduce(lambda a, b: a and b, rsc_constraints)

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
        constraints = []
        constraints += [self.check_resource_constraints() if self.constraints["resource"] else True]
        constraints += [self.check_inter_layer_matching_folding() if self.constraints["inter_layer_matching"] else True]
        for node in self.network.nodes:
            constraints += [self.network.nodes[node]["hw"].check_constraints()]
        # ensure it's within all constraints
        return reduce(lambda a, b: a and b, constraints)

    def optimise(self):
        pass

