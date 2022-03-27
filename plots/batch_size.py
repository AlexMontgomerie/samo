import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import matplotlib
import matplotlib.pyplot as plt

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 27}

# matplotlib.rc('font', **font)

# plt.rcParams.update({'font.size': 22})
# plt.rc('legend', fontsize=27)
# latency = [196000, 900]
# throughput = [20070400, 6400000, 230400]

batch_size = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]
latency = np.array([84219.12,85888.24,89226.48,95902.96,109255.92,135961,189373.68,358431.2,448707.04,762435.3,1216902.08,2153386.16,3894022.32])
throughput = np.array([1.19E-05,2.33E-05,4.48E-05,8.34E-05,0.000146,0.000235,0.000338,0.000357,0.000571,0.000672,0.000841,0.000951,0.00105186])

clk = 100
reconf_time = 82550

width = 0.35

fig, ax = plt.subplots(1)

plt.grid()
# ax.plot(batch_size, throughput)
ax.plot(np.arange(len(batch_size)), throughput*1000000, color="tab:red")
ax.set_ylabel("throughput (img/s)", color="tab:red")

width = 0.35
ax2 = ax.twinx()
ax2.bar(np.arange(len(batch_size)), latency/1000, width, edgecolor="k", color="tab:green")
ax2.set_ylabel("latency (ms)", color="tab:green")
ax2.set_yscale("log")

# add partition numbers
boxes = [
        Rectangle((  -0.5,0),12.5,max(latency/1000)),
        Rectangle((6 -0.5,0),12.5,max(latency/1000)),
        Rectangle((9 -0.5,0),12.5,max(latency/1000)),
        Rectangle((10-0.5,0),12.5,max(latency/1000)),
        Rectangle((11-0.5,0),12.5,max(latency/1000)),
]

# Create patch collection with specified colour/alpha
pc = PatchCollection(boxes, facecolor="tab:blue", alpha=0.2)

# Add collection to axes
ax2.add_collection(pc)
# ax2.bar(batch_size, latency, width)
# ax.set_xscale("log")


# ax.set_yscale("log")
# plt.axis("off")
# plt.legend(bbox_to_anchor =(0.75, 1.15))
fig.tight_layout()
plt.show()

