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

class Network:

    def __init__(self, reference):
        self.reference = reference
        self.partitions = [copy.deepcopy(reference)]
        self.enable_reconf = True

    def eval_latency(self):
        return sum([ partition.eval_latency() for partition in self.partitions ])

    # def eval_latency(self):
    #     return {
    #         "BRAM" : max([ partition.eval_resource()["BRAM"] for partition in self.partitions ]),
    #         "DSP"  : max([ partition.eval_resource()["DSP"] for partition in self.partitions ]),
    #     }[ partition.eval_latency() for partition in self.partitions ])

    def check_constraints(self):
        # check the partitions make up reference design
        # TODO
        # for partition in self.partitions:
        #     print(partition.check_constraints())
        # check constraints of all partitions
        return reduce(lambda a,b: a and b,
                [ partition.check_constraints() for partition in self.partitions ])

    def summary(self):
        # print the summary for each partition
        for index, partition in enumerate(self.partitions):
            print(f"Partition {index}:\n------------\n")
            partition.summary()
        print(f"Network Summary:\n------------\n")
        print(f"Total Latency: {self.eval_latency()}")
        print("\n")

    def split(self, partition_index, nodes):
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
        if self.enable_reconf:
            return list(self.partitions[partition_index].edges)
        else:
            return []

    def merge(self, partitions):
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
        # NOTE assuming partitions are sequential
        merges = []
        for i in range(1,len(self.partitions)):
            merges.append((i-1,i))
        return merges

