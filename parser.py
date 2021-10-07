import pydot
import os
import random
import copy
import onnx
import onnx.utils
import onnx.numpy_helper
import networkx as nx

import onnx_helper

from hardware import Convolution
from hardware import InnerProduct
from hardware import MaxPool
from hardware import ReLU

def remove_node(graph, node):
    prev_nodes = graph.predecessors(node)
    next_nodes = graph.successors(node)
    graph.remove_node(node)
    for prev_node in prev_nodes:
        for next_node in next_nodes:
            graph.add_edge(prev_node,next_node)

def filter_node_types(graph, layer_type):
    remove_nodes = []
    for node in graph.nodes():
        if graph.nodes[node]['type'] == layer_type:
            remove_nodes.append(node)
    for node in remove_nodes:
        remove_node(graph,node)

def build_graph(model):
    # graph structure
    graph = nx.DiGraph()
    # add all nodes from network
    for node in model.graph.node:
        # get name of node
        name = onnx_helper._name(node)
        # add node to graph
        graph.add_node( name, type=node.op_type, hw=None, inputs={} )
    # add all edges from network
    edges = []
    for name in graph.nodes():
        # get node from model
        node = onnx_helper.get_model_node(model, name)
        # add edges into node
        for input_node in node.input:
            # add initializers
            if onnx_helper.get_model_initializer(model, input_node) is not None:
                # get input details
                input_details = onnx_helper.get_model_input(model, input_node)
                # convolution inputs
                if graph.nodes[name]["type"] == "Conv":
                    if len(input_details.type.tensor_type.shape.dim) == 4:
                        graph.nodes[name]['inputs']['weights'] = input_node
                    if len(input_details.type.tensor_type.shape.dim) == 1:
                        graph.nodes[name]['inputs']['bias'] = input_node
                # inner product inputs
                if graph.nodes[name]["type"] == "Gemm":
                    if len(input_details.type.tensor_type.shape.dim) == 2:
                        graph.nodes[name]['inputs']['weights'] = input_node
                    if len(input_details.type.tensor_type.shape.dim) == 1:
                        graph.nodes[name]['inputs']['bias'] = input_node
            # get the input node name
            input_node = onnx_helper._format_name(input_node)
            if input_node != name:
                edges.append((input_node, name))
        # add eges out of node
        for output_node in node.output:
            output_node = onnx_helper._format_name(output_node)
            if output_node in graph.nodes():
                if output_node != name:
                    edges.append((name,output_node))
    # add edges to graph
    for edge in edges:
        graph.add_edge(*edge)
    # return graph
    return graph

def add_hardware(model, graph):
    # iterate over nodes in graph
    for node in model.graph.node:
        # get node name
        name = onnx_helper._name(node)
        # check if node in graph
        if not name in graph.nodes():
            continue
        # Convolution layer
        if graph.nodes[name]['type'] == "Conv":
            # get number of filters
            weights_input = graph.nodes[name]["inputs"]["weights"]
            weights_dim = onnx_helper.get_model_input(model,weights_input)
            filters = int(weights_dim.type.tensor_type.shape.dim[0].dim_value)
            # get node attributes
            attr = onnx_helper._format_attr(node.attribute)
            # default attributes
            attr.setdefault("group", 1)
            attr.setdefault("strides", [1,1])
            attr.setdefault("pads", [0,0,0,0])
            attr.setdefault("dilations", [1,1])
            # create convolution layer hardware
            graph.nodes[name]['hw'] = Convolution(0, 0, 0, filters,
                    attr["kernel_shape"], attr["strides"], attr["group"], attr["pads"])
            continue
        # InnerProduct Layer
        if graph.nodes[name]['type'] == "Gemm":
            # get number of filters
            weights_input = graph.nodes[name]["inputs"]["weights"]
            weights_dim = onnx_helper.get_model_input(model,weights_input)
            filters = int(weights_dim.type.tensor_type.shape.dim[0].dim_value)
            # create inner product layer hardware
            graph.nodes[name]['hw'] = InnerProduct(0, filters)
            continue
        # Pooling layer
        if graph.nodes[name]['type'] == "MaxPool":
            # get node attributes
            attr = onnx_helper._format_attr(node.attribute)
            # default attributes
            attr.setdefault("strides", [1,1])
            attr.setdefault("pads", [0,0,0,0])
            attr.setdefault("dilations", [1,1])
            # create pooling layer hardware
            graph.nodes[name]['hw'] = MaxPool(0, 0, 0,
                    attr["kernel_shape"], attr["strides"], attr["pads"])
            continue
        if graph.nodes[name]['type'] == "Relu":
            # create relu layer hardware
            graph.nodes[name]['hw'] = ReLU(0,0,0)
            continue
        raise NameError(f"{name}: type {str(graph.nodes[name]['type'])} does not exist!")
        print(name,graph.nodes[name]['type'])

def add_dimensions(model, graph):
    # add input dimensions
    input_channels  = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    input_rows      = int(model.graph.input[0].type.tensor_type.shape.dim[2].dim_value)
    input_cols      = int(model.graph.input[0].type.tensor_type.shape.dim[3].dim_value)
    # update input node hardware
    input_node = [ edge for edge, deg in graph.in_degree() if not deg ][0]
    graph.nodes[input_node]['hw'].channels  = input_channels
    graph.nodes[input_node]['hw'].rows      = input_rows
    graph.nodes[input_node]['hw'].cols      = input_cols
    # iterate over layers in model
    nodes = list(graph.nodes())
    nodes.remove(input_node)
    for node in nodes:
        # find previous node
        prev_nodes = graph.predecessors(node)
        for prev_node in prev_nodes: # TODO: support parallel networks
            # get previous node output dimensions
            dim = onnx_helper._out_dim(model, prev_node)
            # update input dimensions
            graph.nodes[node]['hw'].channels = dim[0]
            graph.nodes[node]['hw'].rows     = dim[1]
            graph.nodes[node]['hw'].cols     = dim[2]

def parse_network(filepath):

    # load onnx model
    model = onnx_helper.load(filepath)

    # get graph
    graph = build_graph(model)

    # remove input node
    remove_nodes = []
    for node in graph.nodes:
        if "type" not in graph.nodes[node]:
            remove_nodes.append(node)
    for node in remove_nodes:
        graph.remove_node(node)

    # remove unnecessary nodes
    filter_node_types(graph, "Flatten")

    # add hardware to graph
    add_hardware(model, graph)

    # add layer dimensions
    add_dimensions(model, graph)

    return graph

if __name__ == "__main__":
    graph = parse_network("models/lenet.onnx")
