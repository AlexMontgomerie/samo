import types

from optimiser.brute import BruteForce
from optimiser import Node

import networkx as nx

class ExampleNode(Node):
    def latency_in(self):
        return self.channel_in_folding

    def latency_out(self):
        return self.channel_out_folding

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : (self.channels_in//self.channel_in_folding)*(self.channels_out//self.channel_out_folding),
             "BRAM" : 0,
             "FF" : 0
        }

if __name__ == "__main__":

    # create the computation graph
    graph = nx.DiGraph()
    ## add nodes
    graph.add_node("1", hw=ExampleNode(10,64))
    graph.add_node("2", hw=ExampleNode(64,32))
    graph.add_node("3", hw=ExampleNode(32,16))
    ## add edges
    graph.add_edge(list(graph.nodes)[0], list(graph.nodes)[1])
    graph.add_edge(list(graph.nodes)[1], list(graph.nodes)[2])

    # platform
    platform = {
        "LUT" : 0,
        "DSP" : 20,
        "BRAM" : 0,
        "FF" : 0
    }

    # perform optimisation on the computation graph
    opt = BruteForce(graph, platform)
    opt.optimise()

    # save the configuration

