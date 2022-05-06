from dataclasses import dataclass, field
from functools import reduce
from tabulate import tabulate
from typing import List
import networkx as nx
import json
import copy

from networkx.algorithms.dag import ancestors
from networkx.algorithms.dag import descendants

from .partition import Partition

from fpgaconvnet.models.network.Network import Network as fpgaConvNetNetwork
from fpgaconvnet.models.partition.Partition import Partition as fpgaConvNetPartition

class Network:
    """

    """

    def __init__(self, reference, model_path):
        self.model_path = model_path
        self.reference = reference
        self.partitions = [copy.deepcopy(reference)]
        self.enable_reconf = True
        # fpgaconvnet_net
        self.fpgaconvnet_net = fpgaConvNetNetwork("same", model_path)

    def update_fpgaconvnet(self):
        # set partitions to being empty
        self.fpgaconvnet_net.partitions = []

        # iterate over partitions
        for partition_index in range(len(self.partitions)):
            # get all nodes in partition
            nodes = list(self.partitions[partition_index].nodes)
            # get a subgraph of the network and append it to partitions
            self.fpgaconvnet_net.partitions.append(fpgaConvNetPartition(
                self.fpgaconvnet_net.graph.subgraph(nodes).copy()))
            # update nodes in partition
            for node in nodes:
                self.fpgaconvnet_net.partitions[-1].graph.nodes[node]["hw"] = \
                        self.partitions[partition_index].nodes[node]["hw"].layer

        # update the network
        self.fpgaconvnet_net.update_partitions()

    def eval_latency(self) -> float:
        """
        latency of the network in microseconds
        """
        return sum([ partition.eval_latency()/partition.freq for partition in self.partitions ]) + \
                sum([ partition.platform["reconf_time"] for partition in self.partitions[:-1] ])

    def eval_throughput(self) -> float:
        """
        throughput of the network in mega-samples per second
        """
        return float(self.batch_size)/self.eval_latency()

    def eval_resource(self) -> dict:
        """
        max resource usage across partitions
        """
        rsc = [ p.eval_resource() for p in self.partitions ]
        return {
            "LUT"   : max(rsc, key=lambda x: x["LUT"])["LUT"]/self.partitions[0].platform["resources"]["LUT"]*100,
            "FF"    : max(rsc, key=lambda x: x["FF"])["FF"]/self.partitions[0].platform["resources"]["FF"]*100,
            "DSP"   : max(rsc, key=lambda x: x["DSP"])["DSP"]/self.partitions[0].platform["resources"]["DSP"]*100,
            "BRAM"  : max(rsc, key=lambda x: x["BRAM"])["BRAM"]/self.partitions[0].platform["resources"]["BRAM"]*100
        }

    def eval_cost(self) -> float:
        """
        evaluates either throughput or latency based on the
        objective set by the optimisation method
        """
        if self.objective == "throughput":
            return -self.eval_throughput()
        elif self.objective == "latency":
            return self.eval_latency()

    def check_constraints(self):
        """
        checks all the constraints on the network. This includes:

        - all partition constraints
        - partitions are all a subset of the reference network
        """
        # check the partitions make up reference design
        # TODO
        # check constraints of all partitions
        return reduce(lambda a,b: a and b,
                [ partition.check_constraints() for partition in self.partitions ])

    def summary(self):
        """
        prints out a string to the console giving the current state
        of the network in terms of performance and parameters.
        """
        # print the summary for each partition
        for index, partition in enumerate(self.partitions):
            print(f"Partition {index}:\n------------\n")
            partition.summary()
        print(f"Network Summary:\n------------\n")
        print(f"Objective: {self.objective}")
        print(f"Batch Size: {self.batch_size} (img/batch)")
        print(f"Total Latency: {self.eval_latency()} (us/batch)")
        print(f"Total Throughput: {self.eval_throughput()} (img/us)")
        print("\n")

    def split(self, partition_index, nodes):
        """
        method to perform splits on a given partition in the network.
        The `nodes` in the partition chosen by `partition_index` is
        where the split occurs. The `nodes` must be adjacent for the
        split to happen. This creates two seperate partitions in the
        partition list.

        ```
        self.split(0, (node_0, node_1))
        ```
        """
        # get the node indices
        n0 = nodes[0]
        n1 = nodes[1]
        # check nodes are adjacent
        assert self.reference.has_edge(n0, n1)
        # get partition
        p = self.partitions[partition_index]
        # get all nodes before n0 and n1
        p0_nodes = [*list(ancestors(p, n0)), n0]
        p1_nodes = [n1, *list(descendants(p, n1))]
        # get the new partitions
        p0 = p.subgraph(p0_nodes)
        p1 = p.subgraph(p1_nodes)
        # copy other partition information for new partitions
        p0.platform = p.platform
        p0.freq = p.freq
        p0.wordlength = p.wordlength
        p0.constraints = p.constraints
        p1.platform = p.platform
        p1.freq = p.freq
        p1.wordlength = p.wordlength
        p1.constraints = p.constraints
        # check partitions aren't empty
        assert p0.nodes != None
        assert p1.nodes != None
        # remove partition and add new partitions
        self.partitions.remove(p)
        self.partitions.insert(partition_index, p1)
        self.partitions.insert(partition_index, p0)

    def valid_splits(self, partition_index):
        """
        method to return all valid splits on the partition given
        by `partition_index`. This is a list of pairs of adjacent
        nodes.
        """
        if self.enable_reconf:
            return list(self.partitions[partition_index].edges)
        else:
            return []

    def merge(self, partitions):
        """
        merges two partitions where the output node is adjacent to
        the input node of the other. The function excepts a tuple of
        two partition indices.

        ```
        self.merge((0,1))
        ```
        """
        # get the partitions
        p0 = self.partitions[partitions[0]]
        p1 = self.partitions[partitions[1]]
        # check edge between them exists
        assert self.reference.has_edge(p0.output_node, p1.input_node)
        # remove the 2nd partition
        self.partitions.remove(p1)
        # merge the 2nd partition into first
        self.partitions[partitions[0]] = nx.compose(p0,p1)
        self.partitions[partitions[0]].add_edge(p0.output_node, p1.input_node)
        # copy over partition information
        self.partitions[partitions[0]].freq = p0.freq
        self.partitions[partitions[0]].wordlength = p0.wordlength
        self.partitions[partitions[0]].platform = p0.platform
        self.partitions[partitions[0]].constraints = p0.constraints

    def valid_merges(self):
        """
        returns all valid merges that exist in the network as a
        list of tuples of adjacent partition indices.
        """
        # NOTE assuming partitions are sequential
        merges = []
        for i in range(1,len(self.partitions)):
            merges.append((i-1,i))
        return merges

