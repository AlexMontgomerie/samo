import hls4ml
from tensorflow import keras
# from hls4ml.converters.onnx_to_hls import onnx_to_hls, get_supported_onnx_layers, register_onnx_layer_handler

from .node import HLS4MLWrapper

import networkx as nx

def parse(filepath):

    # create a configuration file
    config = {
        "OnnxModel" : filepath,
        "OutputDir" : "outputs/hls4ml",
        "ProjectName" : "hls4ml_optimiser",
        "XilinxPart" : "xcku115-flvb2104-2-i",
        "ClockPeriod" : 5,
        "IOType" : "io_stream",
        "HLSConfig" : {
            "Model" : {
                "Precision" : "ap_fixed<16,6>",
                "ReuseFactor" : 1
            }
        }
    }

    hls_model = hls4ml.converters.convert_from_config(config)

    # # load keras model
    # model = keras.models.load_model(filepath)

    # # parse the network
    # hls_model = hls4ml.converters.convert_from_keras_model(model)

    # convert into the node wrappers
    network = nx.DiGraph()
    for node in hls_model.graph:
        if type(hls_model.graph[node]) == hls4ml.model.hls_layers.Input:
            continue
        new_node = HLS4MLWrapper(hls_model.graph[node])
        network.add_node(new_node)

    # return the wrapped network
    return network

