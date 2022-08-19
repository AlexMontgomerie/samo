from fpgaconvnet.models.network.Network import Network
from fpgaconvnet.models.partition.Partition import Partition
import fpgaconvnet.tools.graphs as graphs

def export(network, model_path, output_path):

    # parse the model as a Network
    fpgaconvnet_net = Network("samo", model_path)

    # set partitions to being empty
    fpgaconvnet_net.partitions = []

    # iterate over partitions
    for partition_index in range(len(network.partitions)):
        # get all nodes in partition
        nodes = list(network.partitions[partition_index].nodes)
        # get a subgraph of the network and append it to partitions
        fpgaconvnet_net.partitions.append(Partition(
            fpgaconvnet_net.graph.subgraph(nodes).copy()))
        # update nodes in partition
        for node in nodes:
            fpgaconvnet_net.partitions[-1].graph.nodes[node]["hw"] = \
                    network.partitions[partition_index].nodes[node]["hw"].layer

    # update the network
    fpgaconvnet_net.update_partitions()

    # generate the output configuration
    fpgaconvnet_net.save_all_partitions(output_path)
