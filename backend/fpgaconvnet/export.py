import copy

def export(network, model_path):

    # parse the model as a Network
    fpgaconvnet_network = Network("same", model_path)

    # iterate over the nodes of the optimised network, and
    # add them to the fpgaconvnet network
    for node in network.nodes:

    # update the network




