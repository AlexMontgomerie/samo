from finn.custom_op.registry import getCustomOp

from .network import FinnNetworkWrapper
from .node import FinnNodeWrapper

def parse(model):
    # create the computation graph
    network = FinnNetworkWrapper()
    ## add nodes
    for finn_node in model.graph.node:
        channels_in  = model.get_tensor_shape(finn_node.input[0])[-1]
        channels_out = model.get_tensor_shape(finn_node.output[0])[-1]
        network.add_node(finn_node.name, hw=FinnNodeWrapper(getCustomOp(finn_node), channels_in, channels_out))
    ## add edges
    for i in range(len(list(network.nodes))-1):
        network.add_edge(list(network.nodes)[i], list(network.nodes)[i+1])

    assert network.validate()

    return network

