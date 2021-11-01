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

    def optimise(self):
        pass

