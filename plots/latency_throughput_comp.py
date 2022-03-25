import matplotlib
import matplotlib.pyplot as plt

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 27}

# matplotlib.rc('font', **font)

# plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=27)
# latency = [196000, 900]
# throughput = [20070400, 6400000, 230400]

latency = [113571,6422528,802816,401408,12544]
throughput = [57803436,51380224,102760448,51380224,102760448,51380224,102760448,102760448,102760448,102760448,102760448,102760448,1611904]


clk = 100
reconf_time = 82550

labels = ["latency", "throughput"]

fig, ax = plt.subplots()

width = 0.35

def plot_bar(ax, data, offset=0.5, batch_size=1):
    latency_total = 0
    for i in range(len(data)):
        # reconf_time
        if i > 0:
            n = reconf_time/batch_size
            if i == len(data)-1:
                ax.bar(offset, n, width, bottom=latency_total, hatch="xx", color="w", edgecolor="k", label="reconfiguration")
            else:
                ax.bar(offset, n, width, bottom=latency_total, hatch="xx", color="w", edgecolor="k")
            latency_total += n
        n = data[i]/(clk*batch_size)
        # n = data[i]/(clk)
        if i == len(data)-1:
            ax.bar(offset, n, width, bottom=latency_total, color="tab:green", edgecolor="k", label="partition execution")
        else:
            ax.bar(offset, n, width, bottom=latency_total, color="tab:green", edgecolor="k")
        latency_total += n

# plot_bar(ax, latency)
plot_bar(ax, throughput)
# plot_bar(ax, throughput, offset=1.0, batch_size=256)

# ax.set_yscale("log")
plt.axis("off")
# plt.legend(bbox_to_anchor =(0.75, 1.15))
plt.show()

