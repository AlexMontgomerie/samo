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
            for index, layer in enumerate(partition):
                partition.nodes[layer]["hw"].update(hw_update=True)

    def optimise_single_partition(self, partition_index):
        print(f"Partition {partition_index}:\n------------\n")

        if not self.network.partitions[partition_index].check_constraints():
            return False

        step = True
        log = []
        while step:
            step = False
            partition = self.network.partitions[partition_index]

            cost = self.network.eval_cost()
            node_latencys = np.array([ partition.nodes[layer]["hw"].latency() for layer in list(partition.nodes())])

            node_index = np.argsort(node_latencys)[-1]
            layer = list(partition.nodes())[node_index]
            node_hw = partition.nodes[layer]["hw"]

            layer_configurations = list(itertools.product(
                node_hw.valid_channel_in_folding,
                node_hw.valid_channel_out_folding,
                node_hw.valid_kernel_folding))

            current_config = [node_hw.channel_in_folding, node_hw.channel_out_folding, node_hw.kernel_folding]
            layer_configurations = list(filter(lambda x: np.prod(x) > np.prod(current_config), layer_configurations))

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

                for config in layer_configurations:
                    # only explore the next and closest folding
                    if len(next_folding_candidates) > 1:
                        break

                    partition = self.network.partitions[partition_index]

                    network_copy = copy.deepcopy(self.network)
                    node_hw = partition.nodes[layer]["hw"]

                    node_hw.channel_in_folding = config[0]
                    partition.folding_match(layer, config[0], "io")
                    node_hw.channel_out_folding = config[1]
                    partition.folding_match(layer, config[1], "io")
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
                chosen = True
                sorted_candidates = dict(sorted(step_candidates.items(), key=lambda kv: kv[1].partitions[partition_index].avg_rsc_util()))
                for config, network in sorted_candidates.items():
                    if chosen:
                        self.network = network

                    # log the current resources and cost
                    new_cost = self.network.eval_cost()
                    #new_resource = self.network.partitions[partition_index].eval_resource()

                    # update the log
                    if chosen:
                        log += [[
                                time.time()-self.start_time,
                                new_cost,
                                #new_resource["BRAM"],
                                #new_resource["DSP"],
                                #new_resource["LUT"],
                                #new_resource["FF"],
                                #chosen,
                                #layer,
                                #config
                        ]]

                    chosen = False
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

        # write log to a file
        with open("outputs/log.csv".format(partition_index), "a") as f:

            writer = csv.writer(f)
            [ writer.writerow(row) for row in log ]

        partition.summary()

        return True

    def merge_partitions(self):
        print("resolving memory bound partitions")
        reject_list = []

        while True:
            partitions = copy.deepcopy(self.network.partitions)
            cost = self.network.eval_cost()

            merge_prev_candidates = []
            merge_next_candidates = []

            for partition_index, partition in enumerate(self.network.partitions):
                if partition_index != 0 and partition.try_merge_prev and (partition_index-1, partition_index) not in reject_list:
                    merge_prev_candidates.append(partition_index)
                if partition_index != len(self.network.partitions)-1 and partition.try_merge_next and (partition_index, partition_index+1) not in reject_list:
                    merge_next_candidates.append(partition_index)

            merge_total_candidates = merge_prev_candidates + merge_next_candidates
            merge_total_candidates = list(set(merge_total_candidates))

            if len(merge_total_candidates) == 0:
                break

            partition_latencys = [ self.network.partitions[partition_index].eval_latency() for partition_index in merge_total_candidates]
            partition_index    = merge_total_candidates[partition_latencys.index(max(partition_latencys))]

            if partition_index in merge_next_candidates:
                merge_pair = (partition_index, partition_index+1)
            elif partition_index in merge_prev_candidates:
                merge_pair = (partition_index-1, partition_index)

            print(merge_pair)

            self.network.partitions[merge_pair[0]].reset()
            self.network.partitions[merge_pair[1]].reset()
            self.network.merge(merge_pair)
            status = self.optimise_single_partition(merge_pair[0])

            if not status or self.network.eval_cost() >= cost:
                self.network.partitions = partitions
                reject_list.append(merge_pair)
                print("reject")
            else:
                for i, merge in enumerate(reject_list):
                    if merge[0] >= merge_pair[1]:
                        reject_list[i] = (merge[0]-1,merge[1]-1)
                print("accept")

            print("*******")
    def optimise(self):
        if os.path.exists("outputs/log.csv"):
            os.remove("outputs/log.csv")

        for partition_index in range(len(self.network.partitions)):
            self.optimise_single_partition(partition_index)

        self.merge_partitions()
