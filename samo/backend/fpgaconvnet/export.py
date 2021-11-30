from fpgaconvnet_optimiser.models.network.Network import Network

def export(network, model_path, output_path):

    # parse the model as a Network
    fpgaconvnet_net = Network("same", model_path)

    # iterate over the nodes of the optimised network, and
    # add them to the fpgaconvnet network
    for node in network.nodes:
        fpgaconvnet_net.partitions[0].graph.nodes[node]["hw"] = network.nodes[node]["hw"].layer

    # update the network
    fpgaconvnet_net.update_partitions()

    # generate the output configuration
    fpgaconvnet_net.save_all_partitions(output_path)
