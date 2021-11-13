from dataclasses import dataclass, field
from functools import reduce
from tabulate import tabulate
import networkx as nx

class Network(nx.DiGraph):
    platform: dict = {"BRAM": 0, "DSP": 0, "FF": 0, "LUT": 0}
    constraints: dict = {"resource" : True, "inter_layer_matching" : True}

    def eval_latency(self):
        return max([ self.nodes[layer]["hw"].latency() for layer in self.nodes])

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

    def summary(self):
        # get a summary for the whole network
        latency = self.eval_latency()
        resources = self.eval_resource()
        network_summary = tabulate([[
            int(latency),
            f"{resources['DSP']} / {self.platform['DSP']}",
            f"{resources['BRAM']} / {self.platform['BRAM']}",
            f"{resources['LUT']} / {self.platform['LUT']}",
            f"{resources['FF']} / {self.platform['FF']}"
        ]], headers=["Latency (cycles)", "DSP", "BRAM", "LUT", "FF"])
        # get a summary for each layer
        layer_summary = []
        for node in self.nodes:
            layer = self.nodes[node]["hw"]
            layer_summary.append(
                [ node, int(layer.latency()), layer.resource()["DSP"],
                    layer.resource()["BRAM"], layer.resource()["LUT"], layer.resource()["FF"]] )
        layer_summary = tabulate(layer_summary, headers=["Layer", "Latency (cycles)", "DSP", "BRAM", "LUT", "FF"])

        # print the summary
        print("Network Summary:\n----------------\n")
        print(layer_summary)
        print("\n")
        print(network_summary)

def load_from_opt_network(network, opt_network):
    for i, opt_node in enumerate(opt_network.nodes()):
        opt_layer = opt_network.nodes[opt_node]["hw"]
        
        node = list(network.nodes())[i]
        layer = network.nodes[node]["hw"]

        layer.channel_in_folding = opt_layer.channel_in_folding
        layer.channel_out_folding = opt_layer.channel_out_folding
        layer.kernel_folding = opt_layer.kernel_folding
        layer.update()

