from parser import parse_network

import optimiser.minlp

# load the network
net = parse_network("models/lenet.onnx")

# create the platform
platform = {
    "name" : "zedboard",
    "DSP" : 220,
    "BRAM" : 280,
    "FF" : 106400,
    "LUT" : 53200
}

# run the optimiser
optimiser.minlp.optimise(net, platform)

