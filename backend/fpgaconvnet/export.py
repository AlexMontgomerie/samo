from fpgaconvnet_optimiser.models.network.Network import Network
from fpgaconvnet_optimiser.models.partition.Partition import Partition
import fpgaconvnet_optimiser.tools.graphs as graphs

def export(network, model_path, output_path):

    # parse the model as a Network
    fpgaconvnet_net = Network("same", model_path)

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

    # # partition the hardware
    # for partition_index in range(len(network.partitions)-1):
    #     input_node = network.partitions[partition_index].output_node
    #     output_node = network.partitions[partition_index+1].input_node
    #     fpgaconvnet_net.split_horizontal(partition_index, (input_node, output_node))

    # # iterate over the nodes of the optimised network, and
    # # add them to the fpgaconvnet network
    # for partition_index in range(len(network.partitions)):
    #     for node in network.partitions[partition_index].nodes:
    #         fpgaconvnet_net.partitions[partition_index].graph.nodes[node]["hw"] = \
    #                 network.partitions[partition_index].nodes[node]["hw"].layer
    print(len(fpgaconvnet_net.partitions), output_path)

    # update the network
    fpgaconvnet_net.update_partitions()

    # generate the output configuration
    fpgaconvnet_net.save_all_partitions(output_path)
