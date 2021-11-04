import fpgaconvnet_optimiser.tools.parser as parser
from .node import FPGAConvNetWrapper

import networkx as nx

def parse(filepath):

    # parse the network
    _, graph = parser.parse_net(filepath)

    # convert into the node wrappers
    network = nx.DiGraph()
    for node in graph.nodes:
        new_node = FPGAConvNetWrapper(graph.nodes[node]["hw"])
        network.add_node(node, hw=new_node)

    # return the wrapped network
    return network
