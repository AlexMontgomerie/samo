import csv
import copy
import itertools
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from .network import Network

@dataclass
class RuleBased:
    network: Network

    def update(self):
        for index, layer in enumerate(self.network):
            self.network.nodes[layer]["hw"].update(hw_update=True)

    def optimise(self):
        step = True

        def generator():
            while step:
                yield

        log = []

        pbar = tqdm(generator())
        for _ in pbar:
            step = False

            latency = self.network.eval_latency()
            node_latencys = np.array([ self.network.nodes[layer]["hw"].latency() for layer in list(self.network.nodes())])
            #print(node_latencys)
            node_index = np.argsort(node_latencys)[-1]
            layer = list(self.network.nodes())[node_index]
            node_hw = self.network.nodes[layer]["hw"]

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

                for config in layer_configurations:
                    # only explore the next and closest folding
                    if len(next_folding_candidates) > 1:
                        break

                    network_copy = copy.deepcopy(self.network)
                    node_hw = self.network.nodes[layer]["hw"]

                    node_hw.channel_in_folding = config[0]
                    self.network.folding_match(layer, config[0], "io")
                    node_hw.channel_out_folding = config[1]
                    self.network.folding_match(layer, config[1], "io")
                    node_hw.kernel_folding = config[2]

                    # update the network
                    self.update()

                    # check the network is within platform resource constraints
                    if self.network.check_constraints():
                        step_candidates[config] = copy.deepcopy(self.network)
                        next_folding_candidates.append(np.prod(config))
                        next_folding_candidates = list(set(next_folding_candidates))

                    self.network = network_copy
            
                step = len(step_candidates) > 0

                # choose the transformation with minimal resource                
                chosen = True
                sorted_candidates = dict(sorted(step_candidates.items(), key=lambda kv: kv[1].avg_rsc_util()))
                for config, network in sorted_candidates.items():
                    if chosen:
                        self.network = network
                    
                    # log the current resources and latency
                    new_latency = self.network.eval_latency()
                    new_resource = self.network.eval_resource()

                    # update the log
                    log += [[
                            new_latency,
                            new_resource["BRAM"],
                            new_resource["DSP"],
                            new_resource["LUT"],
                            new_resource["FF"],
                            chosen,
                            layer,
                            config
                    ]]

                    chosen = False

        # write log to a file
        with open("outputs/log.csv", "w") as f:
            writer = csv.writer(f)
            [ writer.writerow(row) for row in log ]
