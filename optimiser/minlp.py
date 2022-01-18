from functools import reduce
from dataclasses import dataclass, field
from gekko import GEKKO
import numpy as np
import types

# import scipy.interpolate
from scipy.interpolate import NearestNDInterpolator
import itertools

from sklearn.linear_model import LinearRegression

from .node import Node
from .network import Network

PRIMES = [2,3,5,7,11,13,17]

@dataclass
class MINLP:
    network: Network

    def __post_init__(self):
        # create the constraints programming model
        self.model = GEKKO()

        # set the solver options (APOPT)
        self.model.options.SPECS = 0
        self.model.options.SOLVER = 1

    def update(self):
        for index, layer in enumerate(self.network):
            self.network.nodes[layer]["hw"].update()

    def find_intervals(self, folding_factors, var, primes):
        # empty list
        if folding_factors == [1]:
            return var
        else:
            # iterate over the primes
            for i in range(len(primes)):
                # get the prime
                p = primes[i]
                # update folding factors
                new_folding_factors = list(set([ x//p for x in folding_factors if x%p == 0 ]))
                # see if the folding factors are updated
                if new_folding_factors != folding_factors and new_folding_factors != []:
                    # increment the variable bounds
                    var[i] += 1
                    # calculate the next variable bound
                    return self.find_intervals(new_folding_factors,var,primes)

    def add_linear_folding_variables(self, folding_vals):
        # get primes in the folding vals
        primes = [ x for x in folding_vals if x in PRIMES ]
        # update the variable bounds
        var_bounds = self.find_intervals(folding_vals, [0]*len(primes), primes)
        # create the variables
        var = []
        for i, bound in enumerate(var_bounds):
            var.append(
                self.model.Var(0, lb=0, ub=bound,
                    fixed_initial=False, integer=True))
        # return variables and their relevant primes
        return var, primes

    def eval_folding_variables(self, var, primes, eval=False):
        if eval:
            return int(np.prod([pow(p,int(x.value[0])) for p,x in zip(primes,var)]))
        return np.prod([pow(p,x) for p,x in zip(primes,var)])

    def plot_resource_space(self, layer, rsc="BRAM"):
        # get folding vals (channel in folding)
        folding_vals = layer.valid_channel_in_folding
        print(folding_vals)
        # get the primes
        primes = [ x for x in folding_vals if x in PRIMES ]
        # get the upper bounds
        var_bounds = self.find_intervals(folding_vals, [0]*len(primes), primes)
        print(primes, var_bounds)
        if len(primes) == 2:
            x,y = np.meshgrid(np.arange(var_bounds[0]+1), np.arange(var_bounds[1]+1))
            print("x: ", x)
            print("y: ", y)
            z = np.array(x)
            for index, _ in np.ndenumerate(x):
                layer.channel_in_folding = pow(primes[0],x[index])*pow(primes[1],y[index])
                layer.update()
                z[index] = layer.resource()[rsc]
            fig, (ax1, ax2) = plt.subplots(1,2,subplot_kw={"projection": "3d"})
            ax1.plot_surface(x,y,z, alpha=0.5)
            x = np.arange(var_bounds[0]+1)
            y = np.arange(var_bounds[1]+1)
            z_intr = np.zeros((x.shape[0], y.shape[0]))
            print(z_intr.shape)
            for i in x:
                for j in y:
                    layer.channel_in_folding = pow(primes[0],x[i])*pow(primes[1],y[j])
                    layer.update()
                    z_intr[i,j] = layer.resource()[rsc]
            fn = RegularGridInterpolator((x,y), z_intr)
            x,y = np.meshgrid(x, y)
            for index, _ in np.ndenumerate(x):
                z[index] = fn([x[index],y[index]])[0]
            ax2.plot_surface(x,y,z, alpha=0.5)
            plt.show()
            print("z: ", z)

    # def resource_regression_model_full(self, layer, rsc="DSP"):

    #     # add channel in grid vals
    #     channel_in_folding_bounds = np.array(self.find_intervals(layer.valid_channel_in_folding,
    #             [0]*len(layer.channel_in_folding_primes),
    #             layer.channel_in_folding_primes))
    #     channel_in_folding_grid = list(itertools.product(*[
    #         np.arange(bound+1) for bound in channel_in_folding_bounds ]))

    #     # add channel out grid vals
    #     channel_out_folding_bounds = np.array(self.find_intervals(layer.valid_channel_out_folding,
    #             [0]*len(layer.channel_out_folding_primes),
    #             layer.channel_out_folding_primes))
    #     channel_out_folding_grid = list(itertools.product(*[
    #         np.arange(bound+1) for bound in channel_out_folding_bounds ]))

    #     # add kernel grid vals TODO

    #     # resource values
    #     rsc_data = []
    #     vals = list(itertools.product(channel_in_folding_grid, channel_out_folding_grid))
    #     vals_regression = []
    #     for val in vals:
    #         # get the regression values
    #         vals_regression.append((*val[0], *val[1]))
    #         # get folding values
    #         layer.channel_in_folding = int(self.eval_folding_variables(val[0],layer.channel_in_folding_primes))
    #         layer.channel_out_folding = int(self.eval_folding_variables(val[1],layer.channel_out_folding_primes))
    #         # get the resources for that layer and folding values
    #         rsc_data.append(layer.resource()[rsc])

    #     # perform linear regression
    #     reg = LinearRegression().fit(vals_regression, rsc_data)
    #     print(reg.score(vals_regression, rsc_data))

    def add_bspline_model(self, x_data, y_data, z_data):
        x = self.model.Var()
        y = self.model.Var()
        z = self.model.Var()
        print(x_data, y_data, z_data)
        if len(x_data) == 1 and len(y_data) > 2:
            self.model.pwl(y, z, y_data, z_data, bound_x=False)
        else:
            self.model.bspline(x, y, z, x_data, y_data,np.array(z_data).reshape(len(x_data), len(y_data)) )
        return x, y, z

    def add_3d_sos1_approximation(self, x0_data, x1_data, x2_data, y_data):

        # reshape y data
        y_data = np.array(y_data).reshape(len(x0_data), len(x1_data), len(x2_data))

        # create binary variables
        binary_var = self.model.Array(self.model.Var, y_data.shape, integer=True)
        for i in range(y_data.shape[0]):
            for j in range(y_data.shape[1]):
                for k in range(y_data.shape[2]):
                    binary_var[i,j,k].lower = 0
                    binary_var[i,j,k].upper = 0
                    # binary_var[i,j,k].integer = True

        # create sos1 constraint
        self.model.Equation(self.model.sum(binary_var) == 1)

        # # create the discrete co-ordinates
        x0 = self.model.Intermediate(np.sum(np.tensordot(x0_data, binary_var, axes=0)))
        x1 = self.model.Intermediate(np.sum(np.tensordot(np.swapaxes(binary_var, 1, 0), x1_data, axes=0)))
        x2 = self.model.Intermediate(np.sum(np.tensordot(np.swapaxes(binary_var, 2, 0), x2_data, axes=0)))

        # create the approximation
        y = self.model.Intermediate(np.sum(y_data*binary_var))

        # return intermediates
        return x0, x1, x2, y

    def resource_regression_model(self, layer, rsc="DSP"):

        # resource values
        rsc_data = []
        vals = list(itertools.product(layer.valid_channel_in_folding, layer.valid_channel_out_folding))
        vals_regression = []
        for val in vals:
            # get folding values
            layer.channel_in_folding = val[0]
            layer.channel_out_folding = val[1]
            # get the resources for that layer and folding values
            rsc_data.append(layer.resource()[rsc])

        # perform linear regression
        reg = LinearRegression().fit(vals, rsc_data)

        # return coefficients and intercept
        return reg.coef_, reg.intercept_

    def differentiable_resource_model(self, layer):
        """
        This is based off of heuristics for what the models should be, as well as limitations
        of the optimisation tool.
        """

        rsc_data = {
            "BRAM"  : [],
            "DSP"   : [],
            "LUT"   : [],
            "FF"    : []
        }

        vals = list(itertools.product(layer.valid_channel_in_folding,
            layer.valid_channel_out_folding, layer.valid_kernel_folding))
        # vals = list(itertools.product(layer.valid_channel_in_folding,
        #     layer.valid_channel_out_folding))


        # iterate over valies
        for val in vals:
            # update the folding values
            layer.channel_in_folding = val[0]
            layer.channel_out_folding = val[1]
            # layer.kernel_folding = val[2]
            layer.kernel_folding = 1
            # get the resources for this configuration
            layer_rsc = layer.resource()
            for rsc in layer_rsc:
                rsc_data[rsc].append(layer_rsc[rsc])

        # make the folding factors variables again
        layer.channel_in_folding = self.eval_folding_variables(
                layer.channel_in_folding_var, layer.channel_in_folding_primes)
        layer.channel_out_folding = self.eval_folding_variables(
                layer.channel_out_folding_var, layer.channel_out_folding_primes)
        layer.kernel_folding = self.eval_folding_variables(
                layer.kernel_folding_var, layer.kernel_folding_primes)

        # update the node with variable folding factors
        layer.update()

        # create new models for resources
        ## BRAM
        ### simplify data points
        bram_data = {}
        bram_x_data = []
        bram_y_data = []
        for x, y in zip(vals, rsc_data["BRAM"]):
            bram_data[np.prod(x)] = y
        for x, y in bram_data.items():
            bram_x_data.append(x)
            bram_y_data.append(y)
        ### express bram as a pointwise linear model
        bram_x = self.model.Var()
        bram_x_slack = self.model.Var(lb=0)
        self.model.Equation(bram_x >= layer.channel_in_folding*
                layer.channel_out_folding*layer.kernel_folding)
        layer.bram = self.model.Var()
        self.model.pwl(bram_x, layer.bram, bram_x_data, bram_y_data, bound_x=False)

        # ## DSP
        layer.dsp = self.model.Intermediate(layer.channel_in_folding*layer.channel_out_folding*layer.kernel_folding)

        ## LUT

        lut_x0, lut_x1, lut_x2, lut = self.add_3d_sos1_approximation(
                layer.valid_channel_in_folding, layer.valid_channel_out_folding,
                layer.valid_kernel_folding, rsc_data["LUT"])
        self.model.Equation(lut_x0 == layer.channel_in_folding)
        self.model.Equation(lut_x1 == layer.channel_out_folding)
        self.model.Equation(lut_x2 == layer.kernel_folding)
        layer.lut = self.model.Var()
        self.model.Equation(layer.lut == lut)


        # lut_x_data = layer.valid_channel_in_folding
        # lut_y_data = layer.valid_channel_out_folding



        # lut_x, lut_y, layer.lut = self.add_bspline_model(lut_x_data, lut_y_data, rsc_data["LUT"])
        # self.model.Equation(lut_x >= layer.channel_in_folding)
        # self.model.Equation(lut_y >= layer.channel_out_folding)

        # ## FF
        # ff_x_data = layer.valid_channel_in_folding
        # ff_y_data = layer.valid_channel_out_folding
        # ff_x = self.model.Var(0, lb=0)
        # ff_y = self.model.Var(0, lb=0)
        # self.model.Equation(ff_x == layer.channel_in_folding)
        # self.model.Equation(ff_y == layer.channel_out_folding)
        # layer.ff = self.model.Var(0, lb=0)
        # self.model.bspline(ff_x, ff_y, layer.ff, ff_x_data, ff_y_data,
        #         np.array(rsc_data["FF"]).reshape(len(ff_x_data), len(ff_y_data)) )

        return layer

    def optimise(self):

        # add variables
        ## iterate over the nodes in the network
        for node in self.network:
            ## add channel in folding variables
            var, primes = self.add_linear_folding_variables(
                    self.network.nodes[node]["hw"].valid_channel_in_folding)
            ## convert to normal folding factors for latency and resource evaluation
            self.network.nodes[node]["hw"].channel_in_folding_var = var
            self.network.nodes[node]["hw"].channel_in_folding_primes = primes
            self.network.nodes[node]["hw"].channel_in_folding = self.eval_folding_variables(var, primes)
            ## add channel out folding variables
            var, primes = self.add_linear_folding_variables(
                    self.network.nodes[node]["hw"].valid_channel_out_folding)
            ## convert to normal folding factors for latency and resource evaluation
            self.network.nodes[node]["hw"].channel_out_folding_var = var
            self.network.nodes[node]["hw"].channel_out_folding_primes = primes
            self.network.nodes[node]["hw"].channel_out_folding = self.eval_folding_variables(var, primes)
            ## add fine folding variables
            var, primes = self.add_linear_folding_variables(
                    self.network.nodes[node]["hw"].valid_kernel_folding)
            ## convert to normal folding factors for latency and resource evaluation
            self.network.nodes[node]["hw"].kernel_folding_var = var
            self.network.nodes[node]["hw"].kernel_folding_primes = primes
            self.network.nodes[node]["hw"].kernel_folding = self.eval_folding_variables(var, primes)
            ## update the node with variable folding factors
            self.network.nodes[node]["hw"].update()

        # create a differentiable resource model
        for node in self.network:
            self.network.nodes[node]["hw"] = self.differentiable_resource_model(
                    self.network.nodes[node]["hw"])

        # add constraints
        ## iterate over the nodes in the network
        for node in self.network:
            ## add intra folding matching
            if self.network.nodes[node]["hw"].constraints["matching_intra_folding"]:
                self.model.Equation(
                        self.network.nodes[node]["hw"].channel_in_folding == self.network.nodes[node]["hw"].channel_out_folding)
            ## add inter folding matching
            if self.network.constraints["inter_layer_matching"]:
                for prev_node in self.network.predecessors(node):
                    self.model.Equation(
                        self.network.nodes[prev_node]["hw"].channel_out_folding == self.network.nodes[node]["hw"].channel_in_folding)
                for next_node in self.network.successors(node):
                    self.model.Equation(
                        self.network.nodes[node]["hw"].channel_out_folding == self.network.nodes[next_node]["hw"].channel_in_folding)

        ## resource constraints
        # self.model.Equation(sum([ self.network.nodes[node]["hw"].resource()["DSP"] for
        #     node in self.network]) <= self.network.platform["DSP"])

        self.model.Equation(self.model.sum([ self.network.nodes[node]["hw"].bram for
            node in self.network]) <= self.network.platform["BRAM"])
        self.model.Equation(self.model.sum([ self.network.nodes[node]["hw"].dsp for
            node in self.network]) <= self.network.platform["DSP"])
        # self.model.Equation(sum([ self.network.nodes[node]["hw"].lut for
        #     node in self.network]) <= self.network.platform["LUT"])
        # self.model.Equation(sum([ self.network.nodes[node]["hw"].ff for
        #     node in self.network]) <= self.network.platform["FF"])

        # add the objective
        ## define a latency variable
        latency = self.model.Var(lb=1)

        ## add latency constraint for each layer
        for node in self.network:
            self.model.Equation(latency >= self.network.nodes[node]["hw"].latency())

        # define the objective
        self.model.Minimize(1000000*latency)

        # solve the model
        self.model.solve()

        print("latency: ", latency.value[0])
        print("BRAM:    ", sum([ self.network.nodes[node]["hw"].bram.value[0] for
            node in self.network]))
        print("DSP:     ", sum([ self.network.nodes[node]["hw"].dsp.value[0] for
            node in self.network]))
        # print("LUT:     ", sum([ self.network.nodes[node]["hw"].lut.value[0] for
        #     node in self.network]))
        # print("FF:      ", sum([ self.network.nodes[node]["hw"].ff.value[0] for
        #     node in self.network]))

        # convert variables back to values
        ## iterate over the nodes in the network
        for node in self.network:
            ## channel in folding
            self.network.nodes[node]["hw"].channel_in_folding = self.eval_folding_variables(
                    self.network.nodes[node]["hw"].channel_in_folding_var,
                    self.network.nodes[node]["hw"].channel_in_folding_primes,
                    eval=True
            )
            ## channel out folding
            self.network.nodes[node]["hw"].channel_out_folding = self.eval_folding_variables(
                    self.network.nodes[node]["hw"].channel_out_folding_var,
                    self.network.nodes[node]["hw"].channel_out_folding_primes,
                    eval=True
            )
            ## fine folding
            self.network.nodes[node]["hw"].kernel_folding = self.eval_folding_variables(
                    self.network.nodes[node]["hw"].kernel_folding_var,
                    self.network.nodes[node]["hw"].kernel_folding_primes,
                    eval=True
            )
            ## update the node
            self.network.nodes[node]["hw"].update()
            self.network.nodes[node]["hw"].layer.update()

