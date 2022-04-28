from samo.model import Network

class FinnNetworkWrapper(Network):
    def valid_splits(self, partition_index):
        edges = super().valid_splits(partition_index)
        valid = list(filter(lambda x: self.partitions[partition_index].nodes[x[0]]["hw"].split, edges))

        return valid