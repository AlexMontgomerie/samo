import hls4ml
from tensorflow import keras

from .node import HLS4MLNodeWrapper
from .network import HLS4MLNetworkWrapper
from optimiser import Network

def parse(filepath):

    # load keras model
    model = keras.models.load_model(filepath)

    # parse the network
    hls_model = hls4ml.converters.convert_from_keras_model(model)

    # convert into the node wrappers
    network = Network()
    for node in hls_model.graph:
        if type(hls_model.graph[node]) == hls4ml.model.hls_layers.Input:
            continue
        new_node = HLS4MLNodeWrapper(hls_model.graph[node])
        network.add_node(node, hw=new_node)

    # return the wrapped network
    return network

