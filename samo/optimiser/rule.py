import logging
import csv
import copy
import itertools
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import time
import os

from samo.model import Network

@dataclass
class RuleBased:
    network: Network

    def update(self):
        for partition in self.network.partitions:
            for layer in partition:
                partition.nodes[layer]["hw"].update(hw_update=True)

    def optimise_single_partition(self, partition_index):
        # print(f"Partition {partition_index}:\n------------\n")

        if not self.network.partitions[partition_index].check_constraints():
            return False

        step = True
        while step:
            step = False
            partition = self.network.partitions[partition_index]

            node_latencys = np.array([
                partition.nodes[layer]["hw"].latency() for layer in list(partition.nodes())])

            node_index = np.argsort(node_latencys)[-1]
            layer = list(partition.nodes())[node_index]
            node_hw = partition.nodes[layer]["hw"]

            layer_configurations = list(itertools.product(
                node_hw.valid_channel_in_folding,
                node_hw.valid_channel_out_folding,
                node_hw.valid_kernel_folding))

            current_config = [node_hw.channel_in_folding,
                    node_hw.channel_out_folding, node_hw.kernel_folding]

            layer_configurations = list(filter(
                lambda x: np.prod(x) > np.prod(current_config), layer_configurations))

            if node_hw.constraints["matching_intra_folding"]:
                layer_configurations = list(filter(lambda x: x[0] == x[1], layer_configurations))

            layer_configurations = sorted(layer_configurations, key=lambda x: np.prod(x))

            # uncomment the following code, faster optimiser but worse performance
            #def leq_folding(config):
            #    for i in range(len(config)):
            #        if config[i] < current_config[i]:
            #            return False
            #    return True
            #layer_configurations = list(filter(leq_folding, layer_configurations))

            if len(layer_configurations) > 0:
                step_candidates = {}
                next_folding_candidates = []

                prev_throughput_in = partition.eval_throughput_in()
                prev_throughput_out = partition.eval_throughput_out()
                try_merge_prev = False
                try_merge_next = False

                # iterate over configurations
                for config in layer_configurations:

                    # only explore the next and closest folding
                    if len(next_folding_candidates) > 1:
                        break

                    # get the partition
                    partition = self.network.partitions[partition_index]

                    # get the hardware for the layer
                    network_copy = copy.deepcopy(self.network)
                    node_hw = partition.nodes[layer]["hw"]

                    # update input channel folding
                    logging.info(f"({layer}) input channel folding = {config[0]}")
                    node_hw.channel_in_folding = config[0]
                    partition.folding_match(layer, config[0], "io")

                    # update output channel folding
                    logging.info(f"({layer}) output channel folding = {config[1]}")
                    node_hw.channel_out_folding = config[1]
                    partition.folding_match(layer, config[1], "io")

                    # update output channel folding
                    logging.info(f"({layer}) kernel folding = {config[2]}")
                    node_hw.kernel_folding = config[2]

                    # update the network
                    self.update()

                    # check the network is within platform resource constraints
                    if self.network.check_constraints():
                        step_candidates[config] = copy.deepcopy(self.network)
                        next_folding_candidates.append(np.prod(config))
                        next_folding_candidates = list(set(next_folding_candidates))
                    elif not partition.check_memory_bandwdith_constraint():
                        curr_throughput_in = partition.eval_throughput_in()
                        curr_throughput_out = partition.eval_throughput_out()

                        if curr_throughput_in > prev_throughput_in:
                            try_merge_prev = True
                        if curr_throughput_out > prev_throughput_out:
                            try_merge_next = True

                    self.network = network_copy

                step = len(step_candidates) > 0

                # choose the transformation with minimal resource
                minimal_candidate = list(sorted(step_candidates.items(),
                    key=lambda kv: kv[1].partitions[partition_index].avg_rsc_util()))

                # if a minimal candidate exists, update the network
                if minimal_candidate != []:
                    self.network = minimal_candidate[0][1]

            else:
                try_merge_prev = True
                try_merge_next = True

        partition = self.network.partitions[partition_index]
        if partition.eval_latency()/partition.freq < partition.platform["reconf_time"]:
            partition.try_merge_prev = True
            partition.try_merge_next = True
        else:
            partition.try_merge_prev = try_merge_prev
            partition.try_merge_next = try_merge_next

        # partition.summary()

        return True

    def merge_partitions(self):
        # print("resolving memory bound partitions")
        reject_list = []

        while True:
            partitions = copy.deepcopy(self.network.partitions)
            cost = self.network.eval_cost()

            merge_prev_candidates = []
            merge_next_candidates = []

            for partition_index, partition in enumerate(self.network.partitions):
                if partition_index != 0 and partition.try_merge_prev and \
                        (partition_index-1, partition_index) not in reject_list:
                    merge_prev_candidates.append(partition_index)
                if partition_index != len(self.network.partitions)-1 and \
                        partition.try_merge_next and \
                        (partition_index, partition_index+1) not in reject_list:
                    merge_next_candidates.append(partition_index)

            merge_total_candidates = merge_prev_candidates + merge_next_candidates
            merge_total_candidates = list(set(merge_total_candidates))

            if len(merge_total_candidates) == 0:
                break

            partition_latencys = [
                    self.network.partitions[partition_index].eval_latency() \
                            for partition_index in merge_total_candidates]

            partition_index = merge_total_candidates[
                    partition_latencys.index(max(partition_latencys))]

            # merge current partition with next partition
            if partition_index in merge_next_candidates:
                merge_pair = (partition_index, partition_index+1)
            # merge current partition with previous partition
            elif partition_index in merge_prev_candidates:
                merge_pair = (partition_index-1, partition_index)

            # reset both partitions to a minimal state
            self.network.partitions[merge_pair[0]].reset()
            self.network.partitions[merge_pair[1]].reset()

            # merge partitions
            self.network.merge(merge_pair)

            # optimise the new partition
            status = self.optimise_single_partition(merge_pair[0])

            # only keep if it can merge, and the performance is better
            if not status or self.network.eval_cost() >= cost:
                self.network.partitions = partitions
                reject_list.append(merge_pair)
                logging.info(f"merging {merge_pair[0]} with {merge_pair[1]} rejected")
            else:
                for i, merge in enumerate(reject_list):
                    if merge[0] >= merge_pair[1]:
                        reject_list[i] = (merge[0]-1,merge[1]-1)
                logging.info(f"merging {merge_pair[0]} with {merge_pair[1]} accepted")

    def optimise(self):

        # optimise the single partitions on their own
        for partition_index in tqdm(range(len(self.network.partitions)),
                desc="optimising single partitions"):
            self.optimise_single_partition(partition_index)

        # merge partitions
        self.merge_partitions()
