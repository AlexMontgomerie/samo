import copy
import hls4ml
from tensorflow import keras

from optimiser import Network
from optimiser import Partition

from .node import HLS4MLNodeWrapper
from .network import HLS4MLNetworkWrapper

def parse(filepath):

    # load keras model
    model = keras.models.load_model(filepath)

    # parse the network
    hls_model = hls4ml.converters.convert_from_keras_model(model)

    # convert into the node wrappers
    reference = Partition()
    prev_node = None
    for node in hls_model.graph:
        if type(hls_model.graph[node]) == hls4ml.model.hls_layers.Input:
            continue
        new_node = HLS4MLNodeWrapper(hls_model.graph[node])
        reference.add_node(node, hw=new_node)
        # add edge to graph
        if prev_node != None:
            reference.add_edge(prev_node, node)
        prev_node = node

    # create network from reference design
    network = Network(reference)

    # return the wrapped network
    return network

