from dataclasses import dataclass

from gekko import GEKKO
from gekko.gk_variable import GKVariable

@dataclass
class Layer:
    rows: int
    cols: int
    channels: int
    streams_in: GKVariable
    streams_out: GKVariable

    def latency(self):
        return self.rows*self.cols*self.channels/self.streams_in

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : self.streams_in*self.streams_out,
             "BRAM" : 0,
             "FF" : 0
        }

# create the constraints programming model
model = GEKKO()

# create variables
streams_in = [
    model.Var(1, lb=1, ub=10, integer=True),
    model.Var(1, lb=1, ub=50, integer=True),
]
streams_out = [
    model.Var(1, lb=1, ub=10, integer=True),
    model.Var(1, lb=1, ub=50, integer=True),
]

# create a single layer
layer_0 = Layer(10,10,10, streams_in[0], streams_out[0])
layer_1 = Layer(20,20,50, streams_in[1], streams_out[1])

# add resource constraints
model.Equation(layer_0.resource()["DSP"]+layer_1.resource()["DSP"] <= 10)

# define a variable which is the latency
latency = model.Var(100000, lb=1, integer=True)

# create constraints that make the latency greater than the others
model.Equation(latency - layer_0.latency() >= 0)
model.Equation(latency - layer_1.latency() >= 0)

# define the objective
# model.Minimize(latency)
model.Minimize(model.max2(layer_0.latency(), layer_1.latency()))

model.options.SOLVER = 1
# model.solve()
print(streams_in, streams_out)

model.fix(streams_in[1], val=5)

model.solve()
print(streams_in, streams_out)
