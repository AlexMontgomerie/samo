from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp

from .network import FinnNetworkWrapper
from optimiser import Partition
from .node import FinnNodeWrapper

import os
import json

def generate_finn_config(model_path, platform, freq, wordlength):

    model_zoo = ["simple","lenet","tfc","sfc","lfc","mpcnn","mobilenetv1","cnv","vgg11"]
    for model_name in model_zoo:
        if model_name in model_path:
            break

    config = {
        "model_name" : model_name,
        "brevitas_finn_onnx" : "brevitas_finn" in model_path,
        "weight_width" : wordlength,
        "acc_width": wordlength,
        "device": {"u250":"U250","zc706":"ZC706","zedboard":"Zedboard"}[platform['name']],
        "clock_cycle": int(1000/freq)
    }
    json_path = "../finn/notebooks/samo/config.json"

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_config = json.load(f)
            if config == existing_config:
                return

    with open(json_path, "w") as f:
        json.dump(config,f,indent=2)

    print("generating finn config. Exit")
    exit()

def parse(filepath, platform):
    # create the computation graph
    reference = Partition()
    reference.platform = platform
    reference.freq = 200.0
    reference.wordlength = 4

    generate_finn_config(filepath, reference.platform, reference.freq, reference.wordlength)

    model = ModelWrapper(filepath)

    ## add nodes
    for finn_node in model.graph.node:
        size_in  = model.get_tensor_shape(finn_node.input[0])
        size_out = model.get_tensor_shape(finn_node.output[0])
        reference.add_node(finn_node.name, hw=FinnNodeWrapper(getCustomOp(finn_node), size_in, size_out))
    ## add edges
    for i in range(len(list(reference.nodes))-1):
        reference.add_edge(list(reference.nodes)[i], list(reference.nodes)[i+1])


    # create network from reference design
    network = FinnNetworkWrapper(reference)

    return network

