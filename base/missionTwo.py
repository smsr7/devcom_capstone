import os
import datetime
import logging
import json
import numpy as np
import networkx as nx
import re
from typing import Set

from .network_edge_two import NetworkEdge


class Mission:
    def __init__(self, config_filename: str, eval=False, seed=None):
        """
        Constructor for the Mission object.

        Parameters:
            config_filename: str
                The path to the JSON mission configuration.
            eval: bool
                If True, the start and end nodes are set as defined in the config.
                If False, they will be randomly selected in the reset() method.
        """
        self.eval = eval

        # Set up logging
        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        filename = f"log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_filename = os.path.join(log_dir, filename)
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s %(message)s"
        )
        self.__current_log = ""

        # Load mission configuration from file
        with open(os.path.join("config", config_filename)) as f:
            data = json.load(f)

        # Build NetworkEdge objects
        self.__network_edges: Set[NetworkEdge] = set()
        for edge_dict in data["edges"]:
            edge = NetworkEdge(edge_dict)
            self.__network_edges.add(edge)

        # Mission-level parameters
        mission_data = data["mission"]
        self.__num_scanner_uavs = mission_data.get("num_scanner_uavs", 0)

        # Start/end and vehicle initial positions
        if self.eval:
            self.__start_node = mission_data["start"]
            self.__end_node = mission_data["end"]
            self.__ugv_location = self.__start_node.upper()
            self.__uav_location = self.__start_node.upper()
            self.__scanner_uav_location = self.__start_node.upper()
        else:
            self.__start_node = None
            self.__end_node = None
            self.__ugv_location = None
            self.__uav_location = None
            self.__scanner_uav_location = None

        # Timing parameters
        self.__human_estimate_time = mission_data["human estimate time"]
        self.__ai_estimate_time = mission_data["AI estimate time"]
        self.__ugv_traversal_time = mission_data["UGV traversal time"]
        self.__ugv_clear_time = mission_data["UGV clear time"]
        self.__uav_traversal_time = mission_data["UAV traversal time"]

        self.__total = 0

        # Build networkx graph and interior-nodes list
        self.network = nx.Graph()
        edges = [
            (e.origin, e.destination, getattr(e, "metadata", {}))
            for e in self.__network_edges
        ]
        self.network.add_edges_from(edges)
        self.nodes = list(self.network.nodes())

        m = int(np.sqrt(len(self.nodes)))
        pattern = re.compile(r"\(\s*(0|[1-9]\d{0,2}|1000)\s*,\s*(0|[1-9]\d{0,2}|1000)\s*\)")
        self.interior_nodes = []
        for node in self.nodes:
            match = pattern.search(node)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                if 5 <= x <= m - 5 and 5 <= y <= m - 5:
                    self.interior_nodes.append(node)

    def reset(self):
        """
        Resets the mission.

        - In eval mode, the start and end nodes remain as defined.
        - Otherwise, they will be randomly selected.
        """
        if not self.eval:
            self.__start_node = np.random.choice(self.interior_nodes)
            self.__end_node = np.random.choice(self.interior_nodes)
            while self.__end_node == self.__start_node:
                self.__end_node = np.random.choice(self.nodes)
            # Initialize vehicle positions at the new start
            self.__ugv_location = self.__start_node.upper()
            self.__uav_location = self.__start_node.upper()
            self.__scanner_uav_location = self.__start_node.upper()

    @property
    def network_edges(self) -> Set[NetworkEdge]:
        return self.__network_edges

    @property
    def start_node(self) -> str:
        return self.__start_node

    @property
    def end_node(self) -> str:
        return self.__end_node

    @property
    def ugv_location(self) -> str:
        return self.__ugv_location

    @property
    def uav_location(self) -> str:
        return self.__uav_location

    @property
    def scanner_uav_location(self) -> str:
        return self.__scanner_uav_location

    @property
    def num_scanner_uavs(self) -> int:
        return self.__num_scanner_uavs

    @property
    def total(self) -> int:
        return self.__total

    @property
    def current_log(self) -> str:
        return self.__current_log

    def __increment_total(self, value: int):
        self.__total += value

    def move_ugv(self, destination_node: str):
        """
        Move the UGV to a valid adjacent location and increase the cost

        Params:
            destination_node : str - The destination node for the UGV to move to

        Returns:
            int - 0 if a landmine was found, 1 if a landmine was cleared, and -1 if the UGV could not be moved
        """

        for edge in self.__network_edges:
            if (
                (self.ugv_location != destination_node) and
                (self.ugv_location == edge.origin or self.ugv_location == edge.destination) and
                (destination_node == edge.origin or destination_node == edge.destination)
            ):
                if edge.landmine_present and not edge.landmine_found:
                    self.__increment_total(self.__ugv_traversal_time)
                    edge.landmine_found = True
                    return 0
                elif edge.landmine_present and edge.landmine_found:
                    self.__increment_total(self.__ugv_clear_time)
                    edge.landmine_cleared = True
                    edge.landmine_present = False
                    self.__ugv_location = destination_node
                    
                    return 1
                else:
                    self.__increment_total(self.__ugv_traversal_time)
                    self.__ugv_location = destination_node
                    
                    return 2
        return -1

    def move_uav(self, destination_node: str):
        """
        Move the traversal UAV; mark edge.uav_scanned.
        """
        for edge in self.__network_edges:
            if (
                (self.__uav_location != destination_node)
                and (self.__uav_location in (edge.origin, edge.destination))
                and (destination_node in (edge.origin, edge.destination))
            ):
                self.__increment_total(self.__uav_traversal_time)
                self.__uav_location = destination_node
                edge.uav_scanned = True
                self.__log_message(
                    f"UAV moved to passage {destination_node}. "
                    f"Estimates can now be obtained for edge {edge.origin}, {edge.destination}."
                )
                return edge
        self.__log_message(
            f"UAV could not be moved to passage {destination_node}. "
            "Check if destination is adjacent."
        )
        return None

    def move_scanner_uav(self, destination_node: str):
        """
        Move the scanner UAV; mark edge.scanner_uav_scanned.
        """
        for edge in self.__network_edges:
            if (
                (self.__scanner_uav_location != destination_node)
                and (self.__scanner_uav_location in (edge.origin, edge.destination))
                and (destination_node in (edge.origin, edge.destination))
            ):
                self.__increment_total(self.__uav_traversal_time)
                self.__scanner_uav_location = destination_node
                edge.scanner_uav_scanned = True
                self.__log_message(
                    f"Scanner UAV moved to passage {destination_node}. "
                    f"High-AI estimate available for edge {edge.origin}, {edge.destination}."
                )
                return edge
        self.__log_message(
            f"Scanner UAV could not move to passage {destination_node}. "
            "Check if destination is adjacent."
        )
        return None

    def query_ai(self):
        """
        Query the standard AI on the selected edge.
        """
        if (
            self.selected_edge is not None
            and getattr(self.selected_edge, "uav_scanned", False)
            and not getattr(self.selected_edge, "ai_queried", False)
        ):
            self.selected_edge.ai_queried = True
            self.__increment_total(self.__ai_estimate_time)
            self.__log_message(
                f"AI queried for edge {self.selected_edge.origin}, "
                f"{self.selected_edge.destination}. Estimate: "
                f"{self.selected_edge.ai_estimate}"
            )
            return True

        self.__log_message(
            "AI could not be queried (edge not selected, not scanned, or already queried)."
        )
        return False

    def query_scanner_ai(self):
        """
        Query the high-reliability AI on the selected edge.
        """
        if (
            self.selected_edge is not None
            and getattr(self.selected_edge, "scanner_uav_scanned", False)
            and not getattr(self.selected_edge, "scanner_ai_queried", False)
        ):
            self.selected_edge.scanner_ai_queried = True
            self.__increment_total(self.__ai_estimate_time)
            self.__log_message(
                f"Scanner AI queried for edge {self.selected_edge.origin}, "
                f"{self.selected_edge.destination}. High estimate: "
                f"{self.selected_edge.high_ai_estimate}"
            )
            return True

        self.__log_message(
            "Scanner AI could not be queried (edge not selected, not scanned, or already queried)."
        )
        return False

    def query_human(self):
        """
        Check if there is a selected edge scanned by the UAV, and query the AI if so

        Returns:
            True if the AI was queried
        """

        if self.selected_edge is not None and self.selected_edge.uav_scanned and not self.selected_edge.human_queried:
            self.selected_edge.human_queried = True
            self.__increment_total(self.__human_estimate_time)
            return True
        else:
            return False

    def get_chosen_edge(self, point_a: str, point_b: str):
        """
        Gets a valid chosen edge and update the UI with its information

        Params:
            point_a : str - The start point for the edge to be gotten
            point_b : str - The end point for the edge to be gotten

        Returns:
            edge : NetworkEdge - the edge that was chosen or None otherwise
        """

        for edge in self.network_edges:
            if (
                (point_a == edge.origin or point_a == edge.destination) and
                (point_b == edge.origin or point_b == edge.destination)
            ):
                self.__selected_edge = edge
                self.__log_message("Selected edge %s, %s" % (point_a, point_b))
                return edge
        self.__log_message("Edge %s, %s could not be found" % (point_a, point_b))
        return None

    def __log_message(self, msg: str):
        self.__current_log = msg
        logging.info(msg)
