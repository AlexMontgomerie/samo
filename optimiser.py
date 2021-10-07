from hardware import Layer

def optimise_network(network, platform):

    # create the constraints programming model
    model = GEKKO()

    # set the solver options (APOPT)
    model.options.SOLVER = 1

    # create all the stream in and out variables
    streams_in = [
        model.Var(1, lb=1, ub=layer.channels, integer=True) for
            layer in network ]
    streams_out = [
        model.Var(1, lb=1, ub=layer.channels, integer=True) for
            layer in network ]

    # update the parsed network with these stream variables
    # TODO

    # add a streams in and out constraint for non convolution or inner product layers
    # TODO

    # add the resource constraints
    model.Equation(sum([ layer.resource()["DSP"] for layer in network]) <= platform["DSP"])

    # create the latency objective
    ## define a latency variable
    latency = model.Var(lb=1, integer=True)

    ## add the latency constraint of each layer
    for layer in network:
        model.Equation(latency - layer.latency() >= 0)

    ## define the objective
    model.Minimize(latency)

    # iterate over the number of streams in and out
    for _ in range(len(streams_in)+len(streams_out)):
        model.solve()








