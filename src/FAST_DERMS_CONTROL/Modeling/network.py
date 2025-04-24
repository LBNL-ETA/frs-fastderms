"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import logging
import warnings
from ..common import fastderms, mRID
from typing import List
import numpy as np
from .equipmentClasses import *
from itertools import chain


class Network(fastderms):
    """Class containing network, system parameters, and connectivity data."""

    def __init__(
        self,
        substation_ID,
        lines: List[Line3P],
        DERs: List[DER],
        Loads: List[Load],
        Sbase: float,
        Vset: float = 1,
        Vmin: float = 0.95,
        Vmax: float = 1.05,
        *args,
        **kw_args,
    ) -> None:

        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        super().__init__(*args, **kw_args)

        # Dictionary of Line3P objects
        self.lines = {}
        self.set_lines(lines, reset=True)
        # Dictionary of DER objects
        self.DERs = {}
        self.set_DERs(DERs, reset=True)
        # Dictionary of Load objects
        self.Loads = {}
        self.set_loads(Loads, reset=True)
        # Dictionary of CompositeResource objects
        composite_resources = kw_args.get("composite_resources", [])
        self.CompositeResources = {}
        self.set_composite_resources(composite_resources, reset=True)

        self.Sbase = Sbase
        # Substation Data
        self.substation_ID = substation_ID  # Bus ID of substation head
        self.Vset = Vset  # Voltage at substation substation [pu]
        self.P0 = [0] * 3  # Substation Power for each phase
        self.P0_up_lim = np.inf  # Substation Power Up limit
        self.P0_dn_lim = -np.inf  # Substation Power Down limit

        # Voltage Limits [pu]
        self.set_voltage_limits(Vmin=Vmin, Vmax=Vmax)
        # Reserves
        self.reserves = {}

        self.Prices = kw_args.get("Prices", Price(tz=self.local_tz))

        nodes = kw_args.get("nodes", None)
        # List of BUS IDs in the network
        if nodes is None:
            # List of BUS IDs in the network
            self.nodes = np.unique(
                [[line.to_bus, line.from_bus] for line in self.get_lines()]
            )
            self.nodes.sort()
        else:
            self.nodes = np.unique(nodes).sort()

        # initialize the lines
        self.init_all_status()

        try:
            # Directed graph
            self.build_network_graph()
        except Exception as e:
            self.logger.error(e)
            self.logger.error("Network graph could not be built !")

        self.logger.warning(
            f"Network initialized with Substation ID {self.substation_ID}\nNodes:\n{self.nodes}"
        )

    def get_substation_ID(self):
        return self.substation_ID

    def set_substation_voltage(self, Vset):
        self.Vset = Vset

    def get_substation_voltage(self):
        return self.Vset

    def set_substation_power(self, P0_value, phi_nr: int = None):
        if phi_nr is None:
            self.P0 = P0_value
        else:
            self.P0[phi_nr] = P0_value

    def get_substation_power(self, phi_nr: int = None, ignore_limits: bool = False):
        if phi_nr is None:
            if ignore_limits:
                return self.P0
            else:
                return min(max(self.P0, self.P0_dn_lim), self.P0_up_lim)
        else:
            return self.P0[phi_nr]

    def set_substation_power_lim(self, kind: str = "up", value=None):
        if kind == "up":
            if value is None or np.isnan(value):
                value = np.inf
            self.P0_up_lim = value
        elif kind == "dn":
            if value is None or np.isnan(value):
                value = -np.inf
            self.P0_dn_lim = value
        else:
            warnings.warn(f"{kind} is not recognized as a limit kind")

    def get_substation_power_lim(self, kind: str = "up"):
        if kind == "up":
            return self.P0_up_lim
        elif kind == "dn":
            return self.P0_dn_lim
        else:
            warnings.warn(f"{kind} is not recognized as a limit kind")

    def set_voltage_limits(self, Vmin: float = None, Vmax: float = None):
        if Vmin is not None:
            self.Vmin = Vmin
        if Vmax is not None:
            self.Vmax = Vmax

    def get_voltage_limits(self):
        return self.Vmin, self.Vmax

    def get_system_reserves(self, kind: str = "up"):
        if kind in ["up", "dn"]:
            return self.reserves[kind]
        elif kind in ["up_dist", "dn_dist"]:
            # Additional Case to cover the MPC case
            return self.reserves[kind]
        else:
            return None

    def set_system_reserves(self, kind: str = "up", value=None):
        if kind in ["up", "dn"]:
            self.reserves[kind] = value
        elif kind in ["up_dist", "dn_dist"]:
            # Additional Case to cover the MPC case
            self.reserves[kind] = value
        else:
            warnings.warn(f"{kind} is not recognized as a reserve kind")

    def set_lines(self, lines: List[Line3P], reset: bool = False):
        if reset:
            self.lines.clear()
            self.logger.info("Resetting lines")
        self.logger.debug(f"Adding {len(lines)} lines to network")
        for line in lines:
            self.lines.update({line.getID(): line})

    def get_lines(self, mRIDs: List[mRID] = None) -> List[Line3P]:
        if mRIDs is None:
            my_list = list(self.lines.values())
        else:
            if type(mRIDs) is not list:
                mRIDs = [mRIDs]
            my_list = [self.lines[mRID] for mRID in mRIDs]
        return my_list

    def set_DERs(self, DERs: List[DER], reset: bool = False):
        if reset:
            self.DERs.clear()
            self.logger.info("Resetting DERs")
        self.logger.debug(f"Adding {len(DERs)} DERs to network")
        for der in DERs:
            self.DERs[der.getID()] = der

    def __get_DERs(self, **kw_args):
        mRID = kw_args.get("mRID", None)
        if mRID is None:
            return list(self.DERs.values())
        else:
            try:
                return self.DERs[mRID]
            except:
                self.logger.error(f"No DER with mRID {mRID} found")
                return None

    def get_DERs(self, **kw_args) -> List[DER]:

        mRIDs = kw_args.get("mRIDs", None)
        if mRIDs is None:
            mRIDs = list(self.DERs.keys())
        elif type(mRIDs) is not list:
            mRIDs = [mRIDs]

        DER_types = kw_args.get("DER_types", None)
        if DER_types is None:
            DER_types = self.get_DER_types()
        elif type(DER_types) is not list:
            DER_types = [DER_types]

        my_list = [self.__get_DERs(mRID=mRID) for mRID in mRIDs]

        my_list = [der for der in my_list if der and der.get_type() in DER_types]

        return my_list

    def get_DER_types(self, filterActive: bool = False):
        DERtypes = np.unique(
            [
                der.get_type()
                for der in self.__get_DERs()
                if (der.get_active_status() or not (filterActive))
            ]
        )
        return DERtypes

    def set_loads(self, loads: List[Load], reset: bool = False):
        if reset:
            self.Loads.clear()
            self.logger.info("Resetting loads")
        self.logger.debug(f"Adding {len(loads)} loads to network")
        for load in loads:
            self.Loads[load.getID()] = load

    def get_loads(self, mRIDs: List[mRID] = None) -> List[Load]:

        if mRIDs is None:
            my_list = list(self.Loads.values())
        else:
            my_list = [self.Loads[mRID] for mRID in mRIDs]
        return my_list

    def set_composite_resources(
        self, composite_resources: List[CompositeResource], reset: bool = False
    ):
        if reset:
            self.CompositeResources.clear()
            self.logger.info("Resetting composite resources")
        self.logger.debug(
            f"Adding {len(composite_resources)} composite resources to network"
        )
        for composite_resource in composite_resources:
            self.CompositeResources[composite_resource.getID()] = composite_resource

    def get_composite_resources(
        self, mRIDs: List[mRID] = None
    ) -> List[CompositeResource]:

        if mRIDs is None:
            my_list = list(self.CompositeResources.values())
        else:
            my_list = [self.CompositeResources[mRID] for mRID in mRIDs]
        return my_list

    def get_ressources_bus(self, busID, obj_type: str, **kw_args) -> List[mRID]:
        """
        Return a list of mRIDs for objects of type: obj_type (load, DER) that is connected to bus with ID: busID
        """
        try:
            if obj_type == "load":
                func = Network.get_loads
            elif obj_type == "DER":
                func = Network.get_DERs
            else:
                raise Exception(f"Object type {obj_type} not recognized")

            obj_list = func(self, **kw_args)

            list_mRID = [obj.getID() for obj in obj_list if busID in obj.get_nodes()]

        except Exception as e:
            self.logger.error(e)
            list_mRID = []
            raise (e)

        finally:
            return list_mRID

    def get_network_graph(self, as_bool: bool = False):
        if as_bool:
            return np.asarray(self.network_graph, dtype=bool)
        else:
            return self.network_graph

    def build_network_graph(self):
        """
        Method to build the network graph that is a directional connectivity matrix of the network
        from buses in row
        to buses in column
        """

        # Refresh list of nodes
        self.nodes = np.unique(
            [[line.to_bus, line.from_bus] for line in self.get_lines()]
        )
        self.nodes.sort()

        # Initializing the directed network graph
        nr_nodes = len(self.nodes)
        self.network_graph = np.full((nr_nodes, nr_nodes), None)

        # Undirected Matrix
        undirected_network_graph = np.array(
            [
                np.array(
                    [
                        any(
                            (line.from_bus in [from_node, to_node])
                            and (line.to_bus in [from_node, to_node])
                            and (from_node != to_node)
                            for line in self.get_lines()
                        )
                        for to_node in self.nodes
                    ]
                )
                for from_node in self.nodes
            ]
        )
        # Building the directed network graph
        nodes = self.make_directed_graph(undirected_network_graph, self.substation_ID)

        self.logger.debug(
            f"\nNetwork graph:{self.print_matrix(self.get_network_graph(as_bool = True))}"
        )

        # Update nodes with the new graph
        nodes_to_deactivate = np.setdiff1d(self.nodes, nodes)
        lines_to_deactivate = [
            line
            for line in self.get_lines()
            if line.from_bus in nodes_to_deactivate
            or line.to_bus in nodes_to_deactivate
        ]
        for line in lines_to_deactivate:
            self.logger.debug(f"Deactivating line: {line.mRID}")
            line.set_active_status(False)
        # Update active nodes
        self.set_active_nodes()

    def make_directed_graph(self, undirected_graph, node, **kw_args):
        """
        Recursive function to build a directed graph from an non directed graph, and an entry node
        """
        visited = kw_args.get("visited", [])

        if not visited:
            self.logger.warning(
                f"Building directed graph from undirected graph starting from node {node}"
            )
            self.logger.info(f"{node}")
        else:
            self.logger.debug(f"Node visited so far: {visited}")

        visited.append(node)

        deco_string = kw_args.get("deco_string", "    ↳→→→")
        child_deco_string = "        " + deco_string

        for child in self.get_direct_children_nodes(node, graph=undirected_graph):
            if child not in visited:
                downstream_lines = {
                    line.mRID: line
                    for line in self.get_lines_between_nodes(node, child)
                }
                for line in downstream_lines.values():
                    line.from_bus = node
                    line.to_bus = child
                self.network_graph[self.nodes == node, self.nodes == child] = (
                    downstream_lines
                )

                self.logger.info(f"{deco_string} {child}")
                _ = self.make_directed_graph(
                    undirected_graph,
                    child,
                    visited=visited,
                    deco_string=child_deco_string,
                )
        return visited

    def crawl_graph(self, node, **kw_args):

        deco_string = kw_args.get("deco_string", "  ↳→→→")
        child_deco_string = "        " + deco_string

        additional_info = kw_args.get("additional_info", {})

        external_file = kw_args.get("external_file", False)
        if external_file:
            file_handler = logging.FileHandler("./network_graph.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(file_handler)

        if "deco_string" not in kw_args:
            self.logger.warning(f"Crawling graph starting from node {node}")
            apdx = f" {additional_info.get(node,'missing')}" if additional_info else ""
            self.logger.info(f"{node}{apdx}")

        for child in self.get_direct_children_nodes(node):
            apdx = f" {additional_info.get(child,'missing')}" if additional_info else ""
            self.logger.info(f"{deco_string} {child}{apdx}")
            _ = self.crawl_graph(
                child, deco_string=child_deco_string, additional_info=additional_info
            )

        if external_file:
            self.logger.removeHandler(file_handler)
            file_handler.close()

    def get_prices(self):
        return self.Prices

    def set_prices(self, Prices: Price):
        self.Prices = Prices

    def init_all_status(self):
        """
        Initialize active status of all lines to True
        """
        self.logger.info("Initializing all lines to active")
        for line in self.get_lines():
            line.set_active_status(True)

    def get_switches(self) -> List[Switch3P]:
        # List of Switches
        switches = [
            line
            for line in self.get_lines()
            if line.__class__.__name__ == Switch3P.__name__
        ]
        return switches

    def get_trafos(self) -> List[Trafo3P]:
        # List of Trafos
        trafos = [
            line
            for line in self.get_lines()
            if line.__class__.__name__ == Trafo3P.__name__
        ]
        return trafos

    def get_batteries(self) -> List[Battery]:
        all_battery_kind = [Battery.__name__] + [
            subclass.__name__ for subclass in Battery.__subclasses__()
        ]
        # List of battery-like equipment
        batteries = [
            der for der in self.get_DERs() if der.__class__.__name__ in all_battery_kind
        ]
        return batteries

    def propagate_status(self, lines: List[Line3P] = None):
        """
        Determine active lines and set active status of lines in Network
        """

        # Calculate Downstream Lines of Switches and set status of downstream lines accordingly
        for line in lines:
            # check device kind
            if line.__class__ == Switch3P:
                switch_status = line.get_switch_status()
                status = line.get_active_status() and switch_status
            else:
                status = line.get_active_status()

            childrenLines = self.get_all_downstream_lines(line)
            for child in childrenLines:
                child.set_active_status(status)

        # Update active nodes
        self.set_active_nodes()
        self.logger.debug(f"Active Nodes: {self.get_active_nodes()}")

        # Create a Node-Node Directional Connectivity Matrix for 3-phase System.
        # Nodes can have arbitrary numbers/IDs.
        nodes = self.get_nodes()
        active_nodes = self.get_active_nodes()
        graph = self.get_network_graph(as_bool=True)

        self.CG = np.array(
            [
                (
                    [
                        [
                            int(
                                any(
                                    graph[id_from][id_to]
                                    and line.from_bus in [from_node, to_node]
                                    and line.to_bus in [from_node, to_node]
                                    and from_node != to_node
                                    and (phase in line.phases)
                                    for line in self.get_lines()
                                )
                            )
                            for phase in [0, 1, 2]
                        ]
                        for id_to, to_node in enumerate(nodes)
                        if to_node in active_nodes
                    ]
                )
                for id_from, from_node in enumerate(nodes)
                if from_node in active_nodes
            ]
        )

        self.logger.debug(
            f"CG \n {self.print_matrix(self.CG, row_headers = list(active_nodes), col_headers = list(active_nodes))}"
        )

        # Create a node-to-DER matrix map for each DER where:
        # keys are DER types that are present in the network (e.g., Battery, PV)
        # values are matrices with 3(# of nodes) rows and (# of DER type present) columns. Elements in matrix are allocation factors for each location of the aggregate loads.
        self.CD = {
            der_type: {
                busID: {
                    derID: [
                        self.__get_DERs(mRID=derID).gamma[busID][phase]
                        for phase in [0, 1, 2]
                    ]
                    for derID in self.get_ressources_bus(
                        busID, obj_type="DER", DER_types=der_type
                    )
                }
                for busID in active_nodes
            }
            for der_type in self.get_DER_types(filterActive=True)
        }

        self.logger.debug(f"CD \n {self.print_dict(self.CD)}")

    def get_nodes(
        self,
        idx: int = None,
    ):

        if idx is None:
            return self.nodes
        else:
            try:
                return self.nodes[idx]
            except:
                raise ()

    def get_active_nodes(self):
        return self.active_nodes

    def set_active_nodes(self):
        self.active_nodes = np.unique(
            [
                [line.to_bus, line.from_bus]
                for line in self.get_lines()
                if line.get_active_status()
            ]
        )

    def get_inactive_nodes(self):
        return np.setdiff1d(self.get_nodes(), self.get_active_nodes())

    def get_direct_children_nodes(self, node, **kw_args):
        """
        Create list of direct dependent/downstream nodes of specified node
        """
        graph = kw_args.get("graph", self.get_network_graph(as_bool=True))
        children = self.nodes[graph[self.nodes == node, :].squeeze()]

        return children

    def get_all_children_nodes(
        self, node, includeParent: bool = False, *args, **kw_args
    ):
        """
        Create list of all dependent/downstream nodes of specified node (including current node)
        """

        visited = kw_args.get("visited", [])
        visited.append(node)

        for child in self.get_direct_children_nodes(node, **kw_args):
            if child not in visited:
                self.get_all_children_nodes(
                    child, visited, includeParent=True, *args, **kw_args
                )

        if not includeParent:
            visited.remove(node)

        return visited

    def get_lines_between_nodes(self, node_from, node_to):
        """
        Create list of lines between two nodes
        """
        lines = [
            line
            for line in self.get_lines()
            if any(
                [
                    (line.from_bus == node_from) and (line.to_bus == node_to),
                    (line.from_bus == node_to) and (line.to_bus == node_from),
                ]
            )
        ]
        return lines

    def get_direct_downstream_lines(
        self, line_up: Line3P, includeParent: bool = False
    ) -> List[Line3P]:
        """
        Create Downstream/Dependent Lines List of Specified Line in Network (including current line)
        """

        line_list = self.get_direct_downstream_lines_node(line_up.to_bus)

        if includeParent:
            line_list = [line_up] + line_list

        return line_list

    def get_direct_downstream_lines_node(self, node) -> List[Line3P]:
        """
        Create list of lines downstream of a given node
        """

        line_list = [
            line
            for line in chain.from_iterable(
                dico.values()
                for dico in chain.from_iterable(
                    self.get_network_graph()[self.get_nodes() == node, :]
                )
                if dico is not None
            )
        ]
        return line_list

    def get_direct_upstream_lines_node(self, node) -> List[Line3P]:
        """
        Create list of lines upstream of a given node
        """

        line_list = [
            line
            for line in chain.from_iterable(
                dico.values()
                for dico in chain.from_iterable(
                    self.get_network_graph()[:, self.get_nodes() == node]
                )
                if dico is not None
            )
        ]
        return line_list

    def get_all_downstream_lines(
        self, line_up: Line3P, includeParent: bool = False, **kw_args
    ) -> List[Line3P]:
        """
        Create list of all downstream lines of specified line in Network (including current line)
        """
        visited = kw_args.get("visited", [])

        visited.append(line_up)

        for line in self.get_direct_downstream_lines(line_up, includeParent=False):
            if line not in visited:
                self.get_all_downstream_lines(line, visited=visited, includeParent=True)

        if not includeParent:
            visited.remove(line_up)

        return visited

    def per_unitize(self, lines: List[Line3P], Vbase: float = None, visited: List = []):

        if not lines:
            self.logger.debug("No downstream lines to per unitize")
            return

        for line in lines:
            if line not in visited:
                Vbasenew = line.per_unitize(Vbase)
                visited.append(line)
                children_lines = self.get_direct_downstream_lines(line)
                self.logger.debug(
                    f"Next; Per unitizing lines: {[line.mRID for line in children_lines]}"
                )
                self.per_unitize(children_lines, Vbasenew, visited=visited)

    def set_approxed_imps(self):
        for line in self.get_lines():
            line.impedance_approx()

    def set_DER_status(self):
        """
        Set DER status and corrects load allocation
        """
        active_nodes = set(self.get_active_nodes())
        for DER in self.get_DERs():
            der_buses = set(DER.get_nodes())

            if der_buses.issubset(active_nodes):
                # All the DER buses are active, no change
                DER.set_DER_status(active=True, gamma=DER.gamma)
                self.logger.info(f"DER {DER.getID()} at all its buses: {der_buses}")
            else:
                der_buses.intersection_update(active_nodes)

                if len(der_buses):
                    # Only some buses are active, reallocation required
                    self.logger.debug(f"Old gamma: {DER.gamma}")
                    new_total = sum(
                        [
                            DER.gamma[bus][phase]
                            for phase in [0, 1, 2]
                            for bus in der_buses
                        ]
                    )
                    gamma = {
                        bus: [DER.gamma[bus][phase] / new_total for phase in [0, 1, 2]]
                        for bus in der_buses
                    }
                    DER.set_DER_status(active=True, gamma=gamma)
                    self.logger.info(
                        f"DER {DER.getID()} at a subset of buses {der_buses}"
                    )
                    self.logger.debug(f"New gamma: {gamma}")

                else:
                    # None of the DER buses are active
                    DER.set_DER_status(active=False, gamma={})
                    self.logger.info(f"DER {DER.getID()} not Active {der_buses}")
