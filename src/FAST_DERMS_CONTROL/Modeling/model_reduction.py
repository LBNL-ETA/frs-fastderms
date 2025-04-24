"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ..common import fastderms, mRID
from .network import Network
from .equipmentClasses import *

from typing import List
from itertools import chain


import copy as cp


class Red_Network(Network):
    def __init__(self, orig_network, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        if isinstance(orig_network, Network):
            kw_args.update(
                {
                    "substation_ID": orig_network.get_substation_ID(),
                    "lines": orig_network.get_lines(),
                    "DERs": orig_network.get_DERs(),
                    "Loads": orig_network.get_loads(),
                    "Sbase": orig_network.Sbase,
                    "Vset": orig_network.Vset,
                    "Vmin": orig_network.Vmin,
                    "Vmax": orig_network.Vmax,
                    "Prices": orig_network.get_prices(),
                }
            )

        super().__init__(*args, **kw_args)

        self.nodes_to_keep = kw_args.get("nodes_to_keep", [])
        self.logger.info(f"Nodes to keep:\n{self.nodes_to_keep}")

        self.mapping = {"_": []}

        self.voltage_limits = {}

        self.logger.info("Reduced Network Initialized ")

    def get_essential_nodes(self):
        return self.nodes_to_keep

    def get_non_essential_nodes(self):
        return [node for node in self.nodes if node not in self.nodes_to_keep]

    def remove_line(self, line_ID):
        self.logger.debug(f"Removing line {line_ID}")
        return self.lines.pop(line_ID)

    def replace_line(
        self, lines_to_remove: List[mRID], lines_to_add: List[Line3P] = None
    ):

        # Add all the lines
        if lines_to_add is None:
            # When no line to add, use '_' as dummy line ID in mapping
            lines_to_add = ["_"]
        else:
            self.set_lines(lines_to_add)

        # Remove all the lines
        for line_to_remove in lines_to_remove:
            line_to_remove = self.remove_line(line_to_remove.getID())
            for line_to_add in lines_to_add:
                if line_to_add == "_":
                    line_to_add_ID = "_"
                else:
                    line_to_add_ID = line_to_add.getID()
                self.mapping.update(
                    {
                        line_to_add_ID: self.mapping.get(line_to_add_ID, []).append(
                            line_to_remove
                        )
                    }
                )

        # Update Network graph
        self.build_network_graph()

    def relocate_loads(self, from_node, to_node, **kw_args):

        self.logger.info(f"Relocating loads from {from_node} to {to_node}")

        phases = kw_args.get("phases", [0, 1, 2])

        # Get loads to relocate
        loads_to_relocate = [
            load for load in self.get_loads() if from_node in load.get_nodes()
        ]

        for load in loads_to_relocate:
            # Check phases of from_node
            bus_phase_before = load.get_nodes_phases()
            phases_from_node = bus_phase_before[from_node]

            try:
                phases = list(set(phases) & set(phases_from_node))

                if len(phases) == 0:
                    raise Exception(
                        f"No phases to relocate for load {load.getID()} on node {from_node}"
                    )

                # Relocate the load
                bus_phase_after = bus_phase_before.update({to_node: phases})
                load.set_nodes(bus_phase=bus_phase_after)

            except Exception as e:
                self.logger.error(e)


class model_red_rule(fastderms):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)
        self.logger.info("Rule Initialized")

    def reduce(self, Red_Network: Red_Network):
        try:
            raise NotImplementedError(
                "Model reduction rule must implement reduce method"
            )
        except Exception as e:
            self.logger.error(e)
            raise e
        finally:
            return Red_Network

    def __call__(self, network):
        return self.reduce(network)


class remove_leaves(model_red_rule):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": "REDUCTION RULE - Remove Leaves"})
        super().__init__(*args, **kw_args)

    def reduce(self, redNetwork: Red_Network):
        try:
            self.logger.info("Reducing model by removing leaves")

            for node in redNetwork.get_essential_nodes():
                children_nodes = redNetwork.get_direct_children_nodes(node)
                for child in children_nodes:
                    self.logger.debug(f"Checking if node {child} is a leaf")
                    leaf = np.unique(self._is_leaf(child, [], redNetwork))
                    if len(leaf) != 0:
                        self.logger.info(f"Leaf Detected: Removing nodes {leaf}")
                        for leaf_node in leaf:
                            # Removing lines connecting the leaf
                            lines_to_remove = redNetwork.get_direct_upstream_lines_node(
                                leaf_node
                            )
                            redNetwork.replace_line(lines_to_remove)
                            # Relocating loads to upstream node
                            redNetwork.relocate_loads(leaf_node, node)
        except Exception as e:
            self.logger.error(e)
            raise e
        finally:
            return redNetwork

    def _is_leaf(self, node, visited_nodes, redNetwork):
        if node in redNetwork.get_essential_nodes():
            return []
        else:
            children_nodes = redNetwork.get_direct_children_nodes(node)
            if len(children_nodes) == 0:
                return visited_nodes + [node]
            else:
                return [
                    node
                    for node in chain.from_iterable(
                        [
                            self._is_leaf(child, visited_nodes + [node], redNetwork)
                            for child in children_nodes
                        ]
                    )
                    if node != []
                ]


class remove_empty_buses(model_red_rule):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": "REDUCTION RULE - Remove Empty Buses"})
        super().__init__(*args, **kw_args)

    def reduce(self, redNetwork: Red_Network):
        try:
            self.logger.info(
                "Reducing model by removing empty buses: bus in between two lines without a load or DER"
            )

            for node in redNetwork.get_non_essential_nodes():
                self.logger.debug(f"Checking if node {node} is empty")

                downstream_lines = redNetwork.get_direct_downstream_lines_node(node)
                upstream_lines = redNetwork.get_direct_upstream_lines_node(node)

                if len(downstream_lines) == 1 and len(upstream_lines) == 1:
                    self.logger.info(f"Empty bus Detected: Removing node {node}")

                    lines_to_remove = downstream_lines + upstream_lines

                    # line to add
                    lines_to_add = []

                    from_bus = upstream_lines[0].from_bus
                    to_bus = downstream_lines[0].to_bus
                    phases_upstream = upstream_lines[0].phases
                    phases_downstream = downstream_lines[0].phases
                    phases = list(set(phases_upstream) & set(phases_downstream))

                    redNetwork.replace_line(lines_to_remove)
        except Exception as e:
            self.logger.error(e)
            raise e
        finally:
            return redNetwork
