import networkx as nx
import numpy as np
from collections import deque

from mission import Mission

from timeKeeper import TimeKeeper

CONFIG = "example_scenario.json"


class MissionWrapper:
    def __init__(self):
        self.NO_SECOND_TRIES = True
        self.MAX_NODES = 3
        self.DEPTH = 2
        self.TIMEKEEPER = True

        self.reset()

    def reset(self):
        self.env = Mission(CONFIG)
        self.GOAL = self.env.end_node

        self.TIMES = {"human": self.env._Mission__human_estimate_time,
                      "ai": self.env._Mission__ai_estimate_time,
                      "move_ugv": self.env._Mission__ugv_traversal_time,
                      "clear_mine": self.env._Mission__ugv_clear_time,
                      "move_uav": self.env._Mission__uav_traversal_time}

        if self.TIMEKEEPER:
            self.tk = TimeKeeper(self.TIMES)

        self.time = 0

        self.uav_node = self.env.start_node
        self.ugv_node = self.env.start_node

        self.ai_scanned_edges = {}
        self.human_scanned_edges = {}
        self.traversed_edges = {}

        self.network = nx.Graph()

        edges = [
            (
                edge.origin,
                edge.destination,
                edge.metadata if hasattr(edge, 'metadata') else {'terrain': edge.terrain}
            )
            for edge in self.env.network_edges]

        self.network.add_edges_from(edges)

    def pad_state(self, edges_array, metadata_array):
        """Pads the state with dummy values to match MAX_NODES size."""
        dummy_edge = ("None", "None")
        dummy_metadata = {"terrain": "unknown", "visibility": 0.0}

        while len(edges_array) < self.MAX_NODES:
            edges_array = np.append(edges_array, [dummy_edge], axis=0)

        while len(metadata_array) < self.MAX_NODES:
            metadata_array = np.append(metadata_array, [dummy_metadata], axis=0)

        return edges_array, metadata_array

    def find_shortest_path(self, agent='uav', weight=None):
        try:
            if agent == 'uav':
                current_node = self.uav_node
            elif agent == 'ugv':
                current_node = self.ugv_node
            else:
                raise ValueError("Agent must be 'uav' or 'ugv'")

            path = nx.shortest_path(self.network, source=current_node, target=self.GOAL, weight=weight)
            return path

        except nx.NetworkXNoPath:
            return None

    def get_uav_state(self):
        """
        Returns the UAV's state up to a specified depth, collecting edges, metadata, traversal statuses,
        AI estimates, and human estimates.
        """
        depth = self.DEPTH
        # Initialize visited nodes and the BFS queue
        visited_nodes = set()
        queue = deque()
        current_node = self.uav_node

        # Initialize per-level data lists
        edges_per_level = [[] for _ in range(depth)]
        metadata_per_level = [[] for _ in range(depth)]
        traversal_statuses_per_level = [[] for _ in range(depth)]
        ai_estimates_per_level = [[] for _ in range(depth)]
        human_estimates_per_level = [[] for _ in range(depth)]

        processed_edges = set()

        queue.append((current_node, 0))
        visited_nodes.add(current_node)

        while queue:
            node, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            # Get edges connected to the node
            edges = list(self.network.edges(node, data=True))

            for start, end, metadata in edges:
                # Edge ID (sorted tuple)
                edge_id = tuple(sorted((start, end)))

                # Check if the edge has been processed
                if edge_id in processed_edges:
                    # Duplicate edge encountered, set values to None
                    edges_per_level[current_depth].append(("None", "None"))
                    metadata_per_level[current_depth].append({"terrain": "unknown", "visibility": 0.0})
                    traversal_statuses_per_level[current_depth].append(None)
                    ai_estimates_per_level[current_depth].append(None)
                    human_estimates_per_level[current_depth].append(None)
                    continue
                else:
                    # Mark the edge as processed
                    processed_edges.add(edge_id)

                    # Collect edge data
                    edges_per_level[current_depth].append((start, end))
                    metadata_per_level[current_depth].append(metadata)

                    # Get traversal status
                    status_info = self.traversed_edges.get(edge_id)
                    traversal_status = status_info['status'] if status_info else None
                    traversal_statuses_per_level[current_depth].append(traversal_status)

                    # Get AI and human estimates
                    ai_estimate = self.ai_scanned_edges.get(edge_id)
                    human_estimate = self.human_scanned_edges.get(edge_id)
                    ai_estimates_per_level[current_depth].append(ai_estimate)
                    human_estimates_per_level[current_depth].append(human_estimate)

                # Get the neighbor node
                neighbor = end if start == node else start

                # If neighbor not visited and within depth, add to queue
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    if current_depth + 1 < depth:
                        queue.append((neighbor, current_depth + 1))

        # Pad the arrays at each level
        padded_edges = []
        padded_metadata = []
        padded_traversal_statuses = []
        padded_ai_estimates = []
        padded_human_estimates = []

        for d in range(depth):
            edges = edges_per_level[d]
            metadata = metadata_per_level[d]
            traversal_statuses = traversal_statuses_per_level[d]
            ai_estimates = ai_estimates_per_level[d]
            human_estimates = human_estimates_per_level[d]

            while len(edges) < self.MAX_NODES:
                edges.append(("None", "None"))
                metadata.append({"terrain": "unknown", "visibility": 0.0})
                traversal_statuses.append(None)
                ai_estimates.append(None)
                human_estimates.append(None)

            edges = edges[:self.MAX_NODES]
            metadata = metadata[:self.MAX_NODES]
            traversal_statuses = traversal_statuses[:self.MAX_NODES]
            ai_estimates = ai_estimates[:self.MAX_NODES]
            human_estimates = human_estimates[:self.MAX_NODES]

            padded_edges.extend(edges)
            padded_metadata.extend(metadata)
            padded_traversal_statuses.extend(traversal_statuses)
            padded_ai_estimates.extend(ai_estimates)
            padded_human_estimates.extend(human_estimates)

        edges_array = np.array(padded_edges, dtype=object)
        metadata_array = np.array(padded_metadata, dtype=object)
        traversal_statuses_array = np.array(padded_traversal_statuses, dtype=object)
        ai_estimates_array = np.array(padded_ai_estimates, dtype=object)
        human_estimates_array = np.array(padded_human_estimates, dtype=object)

        adjacent_nodes = []
        for edge in edges_array:
            start_node, end_node = edge
            if start_node == "None" or end_node == "None":
                adjacent_nodes.append(None)
            else:
                neighbor = end_node if start_node == current_node else start_node
                adjacent_nodes.append(neighbor)

        try:
            path = nx.shortest_path(self.network, source=current_node, target=self.GOAL)
        except nx.NetworkXNoPath:
            path_array = np.zeros(self.MAX_NODES, dtype=int)

        if len(path) < 2:
            path_array = np.zeros(self.MAX_NODES, dtype=int)

        next_node = path[1]

        path_array = np.zeros(self.MAX_NODES, dtype=int)
        if next_node in adjacent_nodes:
            idx = adjacent_nodes.index(next_node)
            path_array[idx] = 1
        else:
            path_array = np.zeros(self.MAX_NODES, dtype=int)

        return edges_array, metadata_array, traversal_statuses_array, ai_estimates_array, human_estimates_array, path_array

    def get_ugv_state(self):
        """Returns the UGV's current state with edges, metadata, traversal statuses, and scanned estimates."""
        ugv_edges = list(self.network.edges(self.ugv_node, data=True))

        edges_array = np.array([(start, end) for start, end, _ in ugv_edges], dtype=object)
        metadata_array = np.array([metadata for _, _, metadata in ugv_edges], dtype=object)

        edges_array, metadata_array = self.pad_state(edges_array, metadata_array)

        current_node = self.ugv_node

        # Get traversal status and scanned estimates for each edge
        traversal_statuses = []
        ai_estimates = []
        human_estimates = []

        for edge in edges_array:
            # Handle dummy edges
            if edge[0] == "None" or edge[1] == "None":
                traversal_statuses.append(None)
                ai_estimates.append(None)
                human_estimates.append(None)
                continue

            edge_id = tuple(sorted((edge[0], edge[1])))

            # Get traversal status
            status_info = self.traversed_edges.get(edge_id)
            if status_info:
                traversal_statuses.append(status_info['status'])
            else:
                traversal_statuses.append(None)  # Edge not traversed yet

            # Get scanned estimates
            ai_estimate = self.ai_scanned_edges.get(edge_id)
            human_estimate = self.human_scanned_edges.get(edge_id)

            ai_estimates.append(ai_estimate)
            human_estimates.append(human_estimate)

        traversal_statuses = np.array(traversal_statuses, dtype=object)
        ai_estimates = np.array(ai_estimates, dtype=object)
        human_estimates = np.array(human_estimates, dtype=object)

        adjacent_nodes = []
        for edge in edges_array:
            start_node, end_node = edge
            if start_node == "None" or end_node == "None":
                adjacent_nodes.append(None)
            else:
                # Determine the neighbor node
                neighbor = end_node if start_node == current_node else start_node
                adjacent_nodes.append(neighbor)

        try:
            path = nx.shortest_path(self.network, source=current_node, target=self.GOAL)
        except nx.NetworkXNoPath:
            path_array = np.zeros(self.MAX_NODES, dtype=int)

        if len(path) < 2:
            path_array = np.zeros(self.MAX_NODES, dtype=int)

        next_node = path[1]

        path_array = np.zeros(self.MAX_NODES, dtype=int)
        if next_node in adjacent_nodes:
            idx = adjacent_nodes.index(next_node)
            path_array[idx] = 1
        else:
            path_array = np.zeros(self.MAX_NODES, dtype=int)

        return edges_array, metadata_array, traversal_statuses, ai_estimates, human_estimates, path_array

    def step_uav(self, action):
        action = np.asarray(action)

        # Validate the action array
        if len(action) != 3 * self.MAX_NODES:
            raise ValueError(f"Action array must be of size {3 * self.MAX_NODES}")

        # Find the index where the '1' is located
        action_indices = np.where(action == 1)[0]
        if len(action_indices) != 1:
            return -1

        action_index = action_indices[0]

        if 0 <= action_index < self.MAX_NODES:
            node_action_index = action_index
            action_type = 'move'
        elif self.MAX_NODES <= action_index < 2 * self.MAX_NODES:
            node_action_index = action_index - self.MAX_NODES
            action_type = 'move_scan_ai'
        elif 2 * self.MAX_NODES <= action_index < 3 * self.MAX_NODES:
            node_action_index = action_index - 2 * self.MAX_NODES
            action_type = 'move_scan_human'
        else:
            raise ValueError("Invalid action index")

        # Get the current UAV state
        edges_array, _, _, _, _, _ = self.get_uav_state()

        # Check if the node_action_index is within bounds
        if node_action_index >= len(edges_array):
            raise ValueError("Node action index is out of range")

        # Get the edge corresponding to the action
        edge = edges_array[node_action_index]
        start_node, end_node = edge

        # Determine the destination node (the node that is not the current UAV node)
        current_node = self.uav_node
        if start_node == current_node:
            destination_node = end_node
        elif end_node == current_node:
            destination_node = start_node
        else:
            raise ValueError(f"Edge does not connect to current UAV node: {current_node}")

        # Move the UAV in Mission
        backend_edge = self.env.move_uav(destination_node)
        if backend_edge is None:
            return -1

        self.uav_node = destination_node

        # Perform scanning
        if action_type in ['move_scan_ai', 'move_scan_human']:
            # Set the selected edge in the backend
            self.env.selected_edge = backend_edge

            # Get the edge identifier (sorted tuple to handle undirected edges)
            edge_id = tuple(sorted((start_node, end_node)))

            if self.TIMEKEEPER:
                self.tk.add_tasks("move_uav")

            # Perform scanning based on action type
            if action_type == 'move_scan_ai':
                self.env.query_ai()
                # Retrieve the AI estimate
                ai_estimate = self.env.selected_edge.ai_estimate
                self.ai_scanned_edges[edge_id] = ai_estimate

                if self.TIMEKEEPER:
                    self.tk.add_tasks("ai")
            elif action_type == 'move_scan_human':
                self.env.query_human()
                human_estimate = self.env.selected_edge.human_estimate
                self.human_scanned_edges[edge_id] = human_estimate

                if self.TIMEKEEPER:
                    self.tk.add_tasks("human")

    def step_ugv(self, action):
        action = np.asarray(action)

        # Validate the action array
        if len(action) != self.MAX_NODES:
            raise ValueError(f"Action array must be of size {self.MAX_NODES}")

        # Find the index where the '1' is located
        action_indices = np.where(action == 1)[0]
        if len(action_indices) != 1:
            return -1

        action_index = action_indices[0]

        # Get the current UGV state
        edges_array, _, _, _, _, _ = self.get_ugv_state()

        # Check if the action_index is within bounds
        if action_index >= len(edges_array):
            raise ValueError("Node action index is out of range")

        # Get the edge corresponding to the action
        edge = edges_array[action_index]
        start_node, end_node = edge

        # Determine the destination node (the node that is not the current UGV node)
        current_node = self.ugv_node
        if start_node == current_node:
            destination_node = end_node
        elif end_node == current_node:
            destination_node = start_node
        else:
            raise ValueError(f"Edge does not connect to current UGV node: {current_node}")

        # Attempt to move the UGV
        move_result = self.env.move_ugv(destination_node)

        edge_id = tuple(sorted((start_node, end_node)))

        if move_result == -1:
            return -1
        elif move_result == 0:
            # Landmine detected, UGV returned to original passageway
            self.traversed_edges[edge_id] = {'status': -1}

            if self.NO_SECOND_TRIES:
                # Attempt to move again to clear the landmine
                move_result = self.env.move_ugv(destination_node)
                if move_result == -1:
                    return -1
                elif move_result == 1 or move_result == 2:
                    # UGV moved successfully
                    self.ugv_node = destination_node
                    self.traversed_edges[edge_id] = {'status': 2}
                    if self.TIMEKEEPER:
                        self.tk.add_tasks("move_ugv")
                        self.tk.add_tasks("clear_mine")

                else:
                    # Should not reach here
                    return -1
            else:
                if self.TIMEKEEPER:
                    self.tk.add_tasks("move_ugv")
                # Do not attempt to move again
                pass
        elif move_result == 1:
            # Landmine was cleared
            self.ugv_node = destination_node
            self.traversed_edges[edge_id] = {'status': 2}
            if self.TIMEKEEPER:
                        self.tk.add_tasks("move_ugv")
                        self.tk.add_tasks("clear_mine")

        elif move_result == 2:
            # UGV moved successfully without incidents
            self.ugv_node = destination_node
            if edge_id not in self.traversed_edges:
                self.traversed_edges[edge_id] = {'status': 1}
                if self.TIMEKEEPER:
                        self.tk.add_tasks("move_ugv")
        else:
            return -1

    def step(self, action):
        ''' 
        param::action - array
            index::[0:MaxNodes * 1] = move_uav
            index::[MaxNodes * 1:MaxNodes * 2] = ai
            index::[MaxNodes * 2:MaxNodes * 3] = human
            index::[MaxNodes * 3 :maxnodes * 1] = move_ugv
        returns:: state, done
        '''

        if self.TIMEKEEPER:
            busy = self.tk.get_status()[1]

            task_keys = ['move_uav', 'ai', 'human', 'move_ugv']

            for i, task_key in enumerate(task_keys):

                task_action = action[i * self.MAX_NODES:(i + 1) * self.MAX_NODES]
                if any(task_action) and busy.get(task_key, 0) == 1:
                    return -1, -1  # if any conflict is found

        if any(action[:self.MAX_NODES*3]):
            self.step_uav(action[:self.MAX_NODES*3])

        if any(action[self.MAX_NODES*3:]):
            self.step_ugv(action[self.MAX_NODES*3:len(action)])

        if self.TIMEKEEPER:
            self.tk.activate_pending_tasks()
            #  self.tk.get_next_time()

        _, metadata, traversal, ai, human, path = self.get_ugv_state()

        state = [metadata, traversal, ai, human, path]

        _, metadata, traversal, ai, human, path = self.get_uav_state()

        state.append([metadata, traversal, ai, human, path])

        if self.ugv_node == self.GOAL:
            done = True

        return state, 1
