import csv
import copy
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from .network import Network

@dataclass
class RuleBased:
    network: Network
    valid_variables: list = field(default_factory=lambda: ["channel_in_folding", "channel_out_folding", "kernel_folding"])

    def update(self):
        for index, layer in enumerate(self.network):
            self.network.nodes[layer]["hw"].update()

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

            step_candidates = {}
            for variable in self.valid_variables:
                network_copy = copy.deepcopy(self.network)
                node_hw = self.network.nodes[layer]["hw"]

                if variable == "channel_in_folding":
                    current_folding = node_hw.channel_in_folding
                    folding_feasible = node_hw.valid_channel_in_folding
                elif variable == "channel_out_folding":
                    current_folding = node_hw.channel_out_folding
                    folding_feasible = node_hw.valid_channel_out_folding
                elif variable == "kernel_folding":
                    current_folding = node_hw.kernel_folding 
                    folding_feasible = node_hw.valid_kernel_folding

                folding_feasible = list(filter(lambda x: x > current_folding, folding_feasible))
                if len(folding_feasible) > 0:
                    folding_feasible = sorted(folding_feasible)
                    folding = folding_feasible[0]

                    if variable == "channel_in_folding":
                        node_hw.channel_in_folding = folding
                        self.network.folding_match(layer, folding, "io")
                    elif variable == "channel_out_folding":
                        node_hw.channel_out_folding = folding
                        self.network.folding_match(layer, folding, "io")
                    elif variable == "kernel_folding":
                        node_hw.kernel_folding = folding

                    # update the network
                    self.update()

                    # check the network is within platform resource constraints
                    if self.network.check_constraints():
                        step_candidates[variable] = copy.deepcopy(self.network)

                    self.network = network_copy
            
            step = len(step_candidates) > 0

            # choose the transformation with minimal resource                
            chosen = True
            sorted_candidates = dict(sorted(step_candidates.items(), key=lambda kv: kv[1].avg_rsc_util()))
            for variable, network in sorted_candidates.items():
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
                        variable
                ]]

                chosen = False

        # write log to a file
        with open("outputs/log.csv", "w") as f:
            writer = csv.writer(f)
            [ writer.writerow(row) for row in log ]
