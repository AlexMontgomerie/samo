import copy

import fpgaconvnet.tools.parser as parser

from samo.model import Network
from samo.model import Partition

from .node import FPGAConvNetWrapper

def parse(filepath, platform, batch_size):

    # parse the network
    _, graph = parser.parse_net(filepath)

    # convert into the node wrappers
    reference = Partition()
    reference.platform = platform
    prev_node = None
    for node in graph.nodes:
        new_node = FPGAConvNetWrapper(graph.nodes[node]["hw"], batch_size=batch_size)
        reference.add_node(node, hw=new_node)
        # add edge to graph
        if prev_node != None:
            reference.add_edge(prev_node, node)
        prev_node = node

    # create network from reference design
    network = Network(reference)
    network.batch_size = batch_size

    # return the wrapped network
    return network
