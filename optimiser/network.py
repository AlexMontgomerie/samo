from dataclasses import dataclass, field
from functools import reduce
from tabulate import tabulate
import networkx as nx
import json

class Network(nx.DiGraph):
    freq: float = 100.0
    wordlength: int = 16
    platform: dict = {
            "resources": {
                "BRAM": 0,
                "DSP": 0,
                "FF": 0,
                "LUT": 0
            },
            "bandwidth": 0.0
        }
    constraints: dict = {"resource" : True, "inter_layer_matching" : True}

    def eval_latency(self):
        return max([ self.nodes[layer]["hw"].latency() for layer in self.nodes])

    def eval_throughput_in(self):
        input_node = [ edge for edge, deg in self.in_degree() if not deg ][0]
        return self.nodes[input_node]["hw"].size_in/self.eval_latency() * \
                self.freq * self.wordlength

    def eval_throughput_out(self):
        output_node = [ edge for edge, deg in self.out_degree() if not deg ][0]
        return self.nodes[output_node]["hw"].size_out/self.eval_latency() * \
                self.freq * self.wordlength

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
        rsc_constraints += [resource["BRAM"] <= self.platform["resources"]["BRAM"]]
        rsc_constraints += [resource["FF"]   <= self.platform["resources"]["FF"]]
        rsc_constraints += [resource["LUT"]  <= self.platform["resources"]["LUT"]]
        rsc_constraints += [resource["DSP"]  <= self.platform["resources"]["DSP"]]
        # if network is within constraints, return true
        return reduce(lambda a, b: a and b, rsc_constraints)

    def check_memory_bandwdith_constraint(self):
        return (self.eval_throughput_in() + self.eval_throughput_out()) < 1000*self.platform["bandwidth"]

    def check_constraints(self):
        # check all the constraints (if they are required)
        for node in self.nodes:
            if not self.nodes[node]["hw"].check_constraints():
                return False
        # check resource constraints
        resource_check = self.check_resource_constraints() if self.constraints["resource"] else True
        bandwidth_check = self.check_memory_bandwdith_constraint()
        return resource_check and bandwidth_check

    def avg_rsc_util(self):
        resource = self.eval_resource()
        avg_rsc_utli = 0.25 * (resource["BRAM"] / self.platform["resources"]["BRAM"]) \
                        + 0.25 * (resource["DSP"] / self.platform["resources"]["DSP"]) \
                        + 0.25 * (resource["LUT"] / self.platform["resources"]["LUT"]) \
                        + 0.25 * (resource["FF"] / self.platform["resources"]["FF"])
        return avg_rsc_utli

    def summary(self):
        # get a summary for the whole network
        latency = self.eval_latency()
        resources = self.eval_resource()
        network_summary = tabulate([[
            int(latency),
            f"{resources['DSP']} / {self.platform['resources']['DSP']}",
            f"{resources['BRAM']} / {self.platform['resources']['BRAM']}",
            f"{resources['LUT']} / {self.platform['resources']['LUT']}",
            f"{resources['FF']} / {self.platform['resources']['FF']}"
        ]], headers=["Latency (cycles)", "DSP", "BRAM", "LUT", "FF"])
        # get a summary for each layer
        layer_summary = []
        for node in self.nodes:
            layer = self.nodes[node]["hw"]
            print(layer.channel_in_folding, layer.channel_out_folding, layer.kernel_folding)
            layer_summary.append(
                [ node, int(layer.latency()), layer.resource()["DSP"],
                    layer.resource()["BRAM"], layer.resource()["LUT"], layer.resource()["FF"]] )
        layer_summary = tabulate(layer_summary, headers=["Layer", "Latency (cycles)", "DSP", "BRAM", "LUT", "FF"])
        bandwidth_summary = tabulate([[
            self.eval_throughput_in(),
            self.eval_throughput_out(),
            f"{self.eval_throughput_in()+self.eval_throughput_out()}/{1000*self.platform['bandwidth']}",
        ]], headers=["in (Mbps)", "out (Mbps)", "total (Mbps)"])

        # print the summary
        print("Network Summary:\n----------------\n")
        print(bandwidth_summary)
        print("\n")
        print(layer_summary)
        print("\n")
        print(network_summary)

    def folding_match(self, node, folding, direction):
        node_hw = self.nodes[node]["hw"]

        if node_hw.constraints["matching_intra_folding"]:
            node_hw.channel_in_folding = folding
            node_hw.channel_out_folding = folding

        channel_in_folding = node_hw.channel_in_folding
        channel_out_folding = node_hw.channel_out_folding

        if self.in_degree(node) > 0 and "i" in direction:
            prev_node = list(self.predecessors(node))[0]
            prev_channel_out_folding = self.nodes[prev_node]["hw"].channel_out_folding
            if self.nodes[prev_node]["hw"].constraints["matching_inter_folding"] or \
               self.nodes[prev_node]["hw"].constraints["divisible_inter_folding"] and max(channel_in_folding, prev_channel_out_folding) % min(channel_in_folding, prev_channel_out_folding) != 0:
                self.nodes[prev_node]["hw"].channel_out_folding = channel_in_folding
                self.folding_match(prev_node, channel_in_folding, "i")

        if self.out_degree(node) > 0 and "o" in direction:
            next_node = list(self.successors(node))[0]
            next_channel_in_folding = self.nodes[next_node]["hw"].channel_in_folding
            if node_hw.constraints["matching_inter_folding"] or \
               node_hw.constraints["divisible_inter_folding"] and max(channel_out_folding, next_channel_in_folding) % min(channel_out_folding, next_channel_in_folding) != 0:
                self.nodes[next_node]["hw"].channel_in_folding = channel_out_folding
                self.folding_match(next_node, channel_out_folding, "o")


    def save_config(self, config_path):
        network_config = {}
        for node in self.nodes:
            node_hw = self.nodes[node]["hw"]

            node_config = {}
            node_config["channel_in_folding"] = node_hw.channel_in_folding
            node_config["channel_out_folding"] = node_hw.channel_out_folding
            node_config["kernel_folding"] = node_hw.kernel_folding

            network_config[node] = node_config

        with open(config_path,"w") as f:
            json.dump(network_config,f,indent=2)

    def load_config(self, config_path):
        with open(config_path,"r") as f:
            network_config = json.load(f)

        for node in self.nodes:
            node_hw = self.nodes[node]["hw"]
            node_config = network_config[node]

            node_hw.channel_in_folding = node_config["channel_in_folding"]
            node_hw.channel_out_folding = node_config["channel_out_folding"]
            node_hw.kernel_folding = node_config["kernel_folding"]
            print(node_config)

        if not self.check_constraints():
            print("Loading a config that may violate constraints")

        for node in self.nodes:
            self.nodes[node]["hw"].update()

