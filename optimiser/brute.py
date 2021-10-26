import itertools
import copy
import numpy as np

def eval_latency(network):
    max_latency_in  = max([ layer.latency_in() for layer in network])
    max_latency_out = max([ layer.latency_out() for layer in network])
    return max(max_latency_in, max_latency_out)

def eval_resource(network):
    return {
        "LUT"   : sum([ layer.resource()["LUT"]  for layer in network]),
        "DSP"   : sum([ layer.resource()["DSP"]  for layer in network]),
        "BRAM"  : sum([ layer.resource()["BRAM"] for layer in network]),
        "FF"    : sum([ layer.resource()["FF"]   for layer in network])
    }

def optimise(network, platform):

    # get all the configurations
    configurations = []
    for layer in network:
        configurations.append(list(itertools.product(
            layer.valid_channel_in_folding,
            layer.valid_channel_out_folding,
            layer.valid_kernel_folding)))
    configurations = list(itertools.product(*configurations))

    # track all valid networks
    valid_networks = {}

    # iterate over all the configurations
    for config in configurations:
        # update the network
        for index, layer in enumerate(network):
            layer.channel_in_folding    = config[index][0]
            layer.channel_out_folding   = config[index][1]
            layer.kernel_folding        = config[index][2]
        # evaluate the latency
        latency  = eval_latency(network)
        resource = eval_resource(network)
        # check constraints
        within_bram = resource["BRAM"] <= platform["BRAM"]
        within_ff   = resource["FF"]   <= platform["FF"]
        within_lut  = resource["LUT"]  <= platform["LUT"]
        within_dsp  = resource["DSP"]  <= platform["DSP"]
        # if network is within constraints, log the network and it's latency
        if within_bram and within_ff and within_lut and within_dsp:
            valid_networks[copy.copy(network)] = latency

    # find the network with the lowest latency
    best_network = min(valid_networks, key=valid_networks.get)

    # return the best network
    return best_network

