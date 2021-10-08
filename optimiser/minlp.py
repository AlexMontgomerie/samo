from functools import reduce
from gekko import GEKKO
import numpy as np

def get_factors(n):
    return list(set(reduce(list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

def eval_latency(network):
    max_latency_in  = max([ network.nodes[layer]["hw"].latency_in(eval=True) for layer in network])
    max_latency_out = max([ network.nodes[layer]["hw"].latency_out(eval=True) for layer in network])
    return max(max_latency_in, max_latency_out)

def optimise(network, platform):

    # create the constraints programming model
    model = GEKKO()

    # set the solver options (APOPT)
    model.options.SPECS = 0
    model.options.SOLVER = 1

    # create all the stream in and out variables
    streams_in = [
        model.Var(1, lb=1, ub=network.nodes[layer]["hw"].channels, fixed_initial=False, integer=True) for
            layer in network ]
    streams_out = [
        model.Var(1, lb=1, ub=network.nodes[layer]["hw"].channels_out, fixed_initial=False, integer=True) for
            layer in network ]

    # update the parsed network with these stream variables
    for index, layer in enumerate(network):
        network.nodes[layer]["hw"].streams_in  = streams_in[index]
        network.nodes[layer]["hw"].streams_out = streams_out[index]

    # add a constraint that all streams along an edge must match
    for index in range(len(streams_in)):
        if index != 0:
            model.Equation(streams_in[index] == streams_out[index-1])

    # add the resource constraints
    model.Equation(sum([ network.nodes[layer]["hw"].resource()["DSP"] for layer in network]) <= platform["DSP"])

    # add additional constraints for non convolution or inner product layers
    for layer in network:
        if network.nodes[layer]["type"] not in ["Gemm", "Conv"]:
            # make sure the streams in and out match
            model.Equation(network.nodes[layer]["hw"].streams_in == network.nodes[layer]["hw"].streams_out)

    # create the latency objective
    ## define a latency variable
    latency = model.Var(lb=1, integer=True)

    ## add the latency constraint of each layer
    for layer in network:
        model.Equation(latency - network.nodes[layer]["hw"].latency_in() >= 0)
        model.Equation(latency - network.nodes[layer]["hw"].latency_out() >= 0)

    ## define the objective
    model.Minimize(latency)

    # iterate over the number of streams in and out
    for step in range(len(streams_in)+len(streams_out)):

        # solve the model
        model.solve(disp=True)

        latency = eval_latency(network)

        # streams in and out latency difference
        streams_in_diff = []
        streams_out_diff = []

        streams_in_closest = []
        streams_out_closest = []

        # iterate over all layers
        for index, layer in enumerate(network):

            # get the layer hardware
            layer = network.nodes[layer]["hw"]

            # get all the valid streams in
            valid_streams_in = get_factors(layer.channels)

            # work out if the solved stream in is valid
            streams_in_value = int(streams_in[index].value[0])
            closest_streams_in = min(valid_streams_in, key=lambda x : abs(x-streams_in_value))
            streams_in_closest.append(closest_streams_in)
            streams_in_diff.append(abs(closest_streams_in - streams_in_value))

            # # if there is no difference, then fix the variable
            # if streams_in_diff == 0:
            #     model.fix_initial(streams_in[index], val=streams_in_value)
            #     layer.streams_in = streams_in[index]

            # get all the valid streams out
            valid_streams_out = get_factors(layer.channels_out)

            # work out if the solved stream out is valid
            streams_out_value = int(streams_out[index].value[0])
            closest_streams_out = min(valid_streams_out, key=lambda x : abs(x-streams_out_value))
            streams_out_closest.append(closest_streams_out)
            streams_out_diff.append(abs(closest_streams_out - streams_out_value))

        # get the index of the max
        streams_in_diff_max_index = streams_in_diff.index(max(streams_in_diff))
        streams_out_diff_max_index = streams_out_diff.index(max(streams_out_diff))

        print(streams_in, streams_out)

        if streams_in_diff[streams_in_diff_max_index] >= streams_out_diff[streams_out_diff_max_index]:
            model.Equation(streams_in[streams_in_diff_max_index] == streams_in_closest[streams_in_diff_max_index])
        else:
            model.Equation(streams_out[streams_out_diff_max_index] == streams_out_closest[streams_out_diff_max_index])

        # re-assign streams in and out
        for index, layer in enumerate(network):
            network.nodes[layer]["hw"].streams_in  = streams_in[index]
            network.nodes[layer]["hw"].streams_out = streams_out[index]

        # calculate total number of different streams
        total_invalid_streams = len(np.nonzero(streams_in_diff)[0]) + len(np.nonzero(streams_out_diff)[0])

        # print status of solver
        print(f"[step {step}] latency = {eval_latency(network)}, invalid streams = {total_invalid_streams}")

        if total_invalid_streams == 0:
            break




