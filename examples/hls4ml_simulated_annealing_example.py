import types

from optimiser.annealing import SimulatedAnnealing
import backend.hls4ml.parser as parser

if __name__ == "__main__":

    # parse the example network
    # graph = parser.parse("models/hls4ml_example.keras")
    graph = parser.parse("models/lenet.onnx")

    # platform
    platform = {
        "LUT" : 437200,
        "DSP" : 900,
        "BRAM" : 1090,
        "FF" : 218600
    }

    # perform optimisation on the computation graph
    opt = SimulatedAnnealing(graph, platform)

    print("latency (before): ", opt.eval_latency())
    opt.optimise()

    print("latency (after): ", opt.eval_latency())

