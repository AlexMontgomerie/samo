"""
This part of the repo contains high-level models of Streaming Architecture
building blocks and their connectivity. In particular, there are three levels
of abstraction that are used:

- `samo.model.node.Node`: fundamental building blocks of the accelerator
- `samo.model.partition.Partition`: subgraph of nodes which constitute an FPGA
configuration
- `samo.model.network.Network`: a collection of partitions which create the
functionality of the whole CNN model
"""

from .node import Node
from .network import Network
from .partition import Partition

