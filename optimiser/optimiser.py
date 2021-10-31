from dataclasses import dataclass
import networkx as nx

@dataclass
class Optimiser:
    network: nx.DiGraph
    platform: dict

    def eval_latency(self):
        max_latency_in  = max([ layer.latency_in() for layer in self.network])
        max_latency_out = max([ layer.latency_out() for layer in self.network])
        return max(max_latency_in, max_latency_out)

    def eval_resource(self):
        return {
            "LUT"   : sum([ layer.resource()["LUT"]  for layer in self.network]),
            "DSP"   : sum([ layer.resource()["DSP"]  for layer in self.network]),
            "BRAM"  : sum([ layer.resource()["BRAM"] for layer in self.network]),
            "FF"    : sum([ layer.resource()["FF"]   for layer in self.network])
        }

    def optimise(self):
        pass

