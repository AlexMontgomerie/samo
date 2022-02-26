import fpgaconvnet_optimiser.tools.parser as parser
from .node import FPGAConvNetWrapper
from optimiser import Network
from optimiser import Partition
import copy

def parse(filepath, platform):

    # parse the network
    _, graph = parser.parse_net(filepath)

    # convert into the node wrappers
    reference = Partition()
    reference.platform = platform
    prev_node = None
    for node in graph.nodes:
        new_node = FPGAConvNetWrapper(graph.nodes[node]["hw"])
        reference.add_node(node, hw=new_node)
        # add edge to graph
        if prev_node != None:
            reference.add_edge(prev_node, node)
        prev_node = node

    # create network from reference design
    network = Network(reference)

    # return the wrapped network
    return network
