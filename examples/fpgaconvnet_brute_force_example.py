import types

from optimiser.brute import BruteForce
import backend.fpgaconvnet.parser as parser

if __name__ == "__main__":

    # parse the example network
    graph = parser.parse("models/simple.onnx")

    # platform
    platform = {
        "LUT" : 0,
        "DSP" : 20,
        "BRAM" : 0,
        "FF" : 0
    }

    # perform optimisation on the computation graph
    opt = BruteForce(graph, platform)
    opt.optimise()

    # save the configuration

