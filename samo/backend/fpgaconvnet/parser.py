import fpgaconvnet_optimiser.tools.parser as parser
from .node import FPGAConvNetWrapper
from samo.model import Network

def parse(filepath):

    # parse the network
    _, graph = parser.parse_net(filepath)

    # convert into the node wrappers
    network = Network()
    for node in graph.nodes:
        new_node = FPGAConvNetWrapper(graph.nodes[node]["hw"])
        network.add_node(node, hw=new_node)

    # return the wrapped network
    return network
