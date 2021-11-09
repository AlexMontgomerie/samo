from optimiser import Network

class HLS4MLNetworkWrapper(Network):

    def eval_latency(self):
        return sum([ self.nodes[layer]["hw"].latency() for layer in self.nodes])

