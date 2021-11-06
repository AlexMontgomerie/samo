from dataclasses import dataclass, field
from functools import reduce
import networkx as nx

class Network(nx.DiGraph):
    platform: dict = {"BRAM": 0, "DSP": 0, "FF": 0, "LUT": 0}
    constraints: dict = {"resource" : True, "inter_layer_matching" : True}

    def eval_latency(self):
        max_latency_in  = max([ self.nodes[layer]["hw"].latency_in() for layer in self.nodes])
        max_latency_out = max([ self.nodes[layer]["hw"].latency_out() for layer in self.nodes])
        return max(max_latency_in, max_latency_out)

    def eval_resource(self):
        return {
            "LUT"   : sum([ self.nodes[layer]["hw"].resource()["LUT"]  for layer in self.nodes]),
            "DSP"   : sum([ self.nodes[layer]["hw"].resource()["DSP"]  for layer in self.nodes]),
            "BRAM"  : sum([ self.nodes[layer]["hw"].resource()["BRAM"] for layer in self.nodes]),
            "FF"    : sum([ self.nodes[layer]["hw"].resource()["FF"]   for layer in self.nodes])
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

    def check_constraints(self):
        # check all the constraints (if they are required)
        constraints = []
        constraints += [self.check_resource_constraints() if self.constraints["resource"] else True]
        for node in self.nodes:
            constraints += [self.nodes[node]["hw"].check_constraints()]
        # ensure it's within all constraints
        return reduce(lambda a, b: a and b, constraints)

