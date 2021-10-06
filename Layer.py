
from ortools.sat.python.cp_model import IntVar

@dataclass
class Layer:
    rows:
    cols:
    channels:
    input_parallelism: IntVar
    output_parallelism: IntVar

    def resource(self):
        return {
             "LUT" : 0,
             "DSP" : 0,
             "BRAM" : 0,
             "FF" : 0
        }



