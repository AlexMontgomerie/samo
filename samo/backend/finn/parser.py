import os
import json

from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp

from samo.model import Partition

from .network import FinnNetworkWrapper
from .node import FinnNodeWrapper

def generate_finn_config(model_path, platform, freq, wordlength, batch_size):

    model_zoo = ["simple","lenet","tfc","sfc","lfc","mpcnn","mobilenetv1","cnv","resnet50"]

    bFound = False
    for model_name in model_zoo:
        if model_name+"_pre_optimiser" in model_path:
            bFound = True
            break
    assert bFound

    config = {
        "model_name" : model_name,
        "brevitas_finn_onnx" : "brevitas_finn" in model_path,
        "weight_width" : wordlength,
        "acc_width": wordlength,
        "device": platform['name'],
        "clock_cycle": int(1000/freq),
        "batch_size": batch_size
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

def parse(filepath, platform, batch_size):
    # create the computation graph
    reference = Partition()
    reference.platform = platform
    reference.freq = 200.0 if platform['name'] == "U250" else 100.0
    reference.wordlength = 4

    generate_finn_config(filepath, reference.platform, reference.freq, reference.wordlength, batch_size)

    model = ModelWrapper(filepath)

    edges = []
    ## add nodes
    for finn_node in model.graph.node:
        size_in  = model.get_tensor_shape(finn_node.input[0])
        size_out = model.get_tensor_shape(finn_node.output[0])
        reference.add_node(finn_node.name, hw=FinnNodeWrapper(getCustomOp(finn_node), size_in, size_out))

        for i in finn_node.input:
            prev_node = model.find_producer(i)
            if prev_node != None:
                edges.append((prev_node.name, finn_node.name))

    ## add edges
    for edge in edges:
        reference.add_edge(*edge)

    for layer in reference.nodes:
        if reference.nodes[layer]['hw'].finn_node.onnx_node.op_type == "StreamingFCLayer_Batch" and reference.out_degree(layer) > 0:
            next_node = list(reference.successors(layer))[0]
            if reference.nodes[next_node]['hw'].finn_node.onnx_node.op_type == "AddStreams_Batch":
                for prev_node in reference.predecessors(next_node):
                    reference.nodes[prev_node]['hw'].constraints["matching_inter_folding"] = True


    # create network from reference design
    network = FinnNetworkWrapper(reference)
    network.batch_size = batch_size

    return network

