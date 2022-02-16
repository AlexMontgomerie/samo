from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp

from .network import FinnNetworkWrapper
from .partition import FinnPartitionWrapper
from .node import FinnNodeWrapper

def parse(filepath):
    model = ModelWrapper(filepath)

    # create the computation graph
    reference = FinnPartitionWrapper()
    ## add nodes
    for finn_node in model.graph.node:
        size_in  = model.get_tensor_shape(finn_node.input[0])
        size_out = model.get_tensor_shape(finn_node.output[0])
        reference.add_node(finn_node.name, hw=FinnNodeWrapper(getCustomOp(finn_node), size_in, size_out))
    ## add edges
    for i in range(len(list(reference.nodes))-1):
        reference.add_edge(list(reference.nodes)[i], list(reference.nodes)[i+1])

    assert reference.validate()

    # create network from reference design
    network = FinnNetworkWrapper(reference)

    return network

