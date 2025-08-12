import networkx as nx
import numpy as np
from collections import deque
from base.mission import Mission
from timeKeeper import TimeKeeper


class MissionWrapper:
    def __init__(self, max_nodes=3, uav_depth=2, ugv_depth=3, config_path="network.json", timekeeper=False):
        """
        Initializes the MissionWrapper environment with default configuration.
        """
        self.config_path = config_path
        self.NO_SECOND_TRIES = True
        self.MAX_NODES = max_nodes
        self.uav_depth = uav_depth
        self.ugv_depth = ugv_depth
        self.TIMEKEEPER = timekeeper

    def reset(self):
        """
        Resets the environment to its initial state and sets up network and mission properties.
        """
        self.env = Mission(self.config_path)
        self.GOAL = self.env.end_node

        # Time-related parameters
        self.TIMES = {
            "human": self.env._Mission__human_estimate_time,
            "ai": self.env._Mission__ai_estimate_time,
            "move_ugv": self.env._Mission__ugv_traversal_time,
            "clear_mine": self.env._Mission__ugv_clear_time,
            "move_uav": self.env._Mission__uav_traversal_time
        }

        if self.TIMEKEEPER:
            self.tk = TimeKeeper(self.TIMES)  # Ensure TimeKeeper class is available

        self.time = 0
        self.uav_node = self.env.start_node
        self.ugv_node = self.env.start_node

        self.last_ugv_edge = None
        self.last_uav_edge = None

        # Tracking traversal and scanning
        self.ai_scanned_edges = {}
        self.human_scanned_edges = {}
        self.uav_scanned_edges = {}   # To track UAV scanning statuses
        self.ugv_traversed_edges = {}  # To track UGV traversal statuses

        # Initialize network
        self.network = nx.Graph()
        edges = [
            (
                edge.origin,
                edge.destination,
                edge.metadata if hasattr(edge, 'metadata') else {'terrain': edge.terrain}
            )
            for edge in self.env.network_edges
        ]
        self.network.add_edges_from(edges)

        state = self._get_state()

        return state

    def pad_state(self, edges_array, metadata_array):
        """
        Pads the state with dummy values to match MAX_NODES size.
        """
        dummy_edge = ("None", "None")
        dummy_metadata = {"terrain": "unknown", "visibility": 0.0}

        while len(edges_array) < self.MAX_NODES:
            edges_array = np.append(edges_array, [dummy_edge], axis=0)

        while len(metadata_array) < self.MAX_NODES:
            metadata_array = np.append(metadata_array, [dummy_metadata], axis=0)

        return edges_array, metadata_array

    def find_shortest_path(self, agent='uav', weight=None):
        """
        Finds the shortest path from the agent's current node to the goal.
        """
        try:
            current_node = self.uav_node if agent == 'uav' else self.ugv_node
            path = nx.shortest_path(self.network, source=current_node, target=self.GOAL, weight=weight)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_uav_state(self):
        """
        Returns the UAV's state up to a specified depth.
        """
        return self._get_agent_state(self.uav_node, self.uav_depth, agent='uav')

    def get_ugv_state(self):
        """
        Returns the UGV's current state with edges, metadata, traversal statuses, and scanned estimates.
        """
        return self._get_agent_state(self.ugv_node, self.ugv_depth, agent='ugv')

    def _get_agent_state(self, current_node, depth, agent='uav'):
        """
        Helper function to retrieve state for UAV or UGV.
        """
        visited_nodes = set()
        queue = deque()

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

                    # Get traversal status based on the agent
                    if agent == 'ugv':
                        status_info = self.ugv_traversed_edges.get(edge_id)
                        traversal_status = status_info['status'] if status_info else None
                    elif agent == 'uav':
                        traversal_status = self.uav_scanned_edges.get(edge_id)
                    else:
                        traversal_status = None

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
            # Get data at level d
            edges = edges_per_level[d]
            metadata = metadata_per_level[d]
            traversal_statuses = traversal_statuses_per_level[d]
            ai_estimates = ai_estimates_per_level[d]
            human_estimates = human_estimates_per_level[d]

            # Pad the data
            while len(edges) < self.MAX_NODES:
                edges.append(("None", "None"))
                metadata.append({"terrain": "unknown", "visibility": 0.0})
                traversal_statuses.append(None)
                ai_estimates.append(None)
                human_estimates.append(None)

            # Truncate to MAX_NODES if needed
            edges = edges[:self.MAX_NODES]
            metadata = metadata[:self.MAX_NODES]
            traversal_statuses = traversal_statuses[:self.MAX_NODES]
            ai_estimates = ai_estimates[:self.MAX_NODES]
            human_estimates = human_estimates[:self.MAX_NODES]

            # Append to the padded lists
            padded_edges.extend(edges)
            padded_metadata.extend(metadata)
            padded_traversal_statuses.extend(traversal_statuses)
            padded_ai_estimates.extend(ai_estimates)
            padded_human_estimates.extend(human_estimates)

        # Convert to numpy arrays
        edges_array = np.array(padded_edges, dtype=object)
        metadata_array = np.array(padded_metadata, dtype=object)
        traversal_statuses_array = np.array(padded_traversal_statuses, dtype=object)
        ai_estimates_array = np.array(padded_ai_estimates, dtype=object)
        human_estimates_array = np.array(padded_human_estimates, dtype=object)

        # Get adjacent nodes
        adjacent_nodes = []
        for edge in edges_array:
            start_node, end_node = edge
            if start_node == "None" or end_node == "None":
                adjacent_nodes.append(None)
            else:
                neighbor = end_node if start_node == current_node else start_node
                adjacent_nodes.append(neighbor)

        # Determine the path array
        try:
            path = nx.shortest_path(self.network, source=current_node, target=self.GOAL)
            next_node = path[1] if len(path) > 1 else None
        except nx.NetworkXNoPath:
            next_node = None

        path_array = np.zeros(self.MAX_NODES, dtype=int)
        if next_node and next_node in adjacent_nodes:
            idx = adjacent_nodes.index(next_node)
            path_array[idx] = 1

        move_array = []

        for edge in edges_array:
            start_node, end_node = edge
            if start_node == "None" or end_node == "None":
                # Dummy edge; cannot move
                move_array.append(0)
            elif current_node == start_node or current_node == end_node:
                # Agent can move along this edge
                move_array.append(1)
            else:
                # Current node not in edge; cannot move
                move_array.append(0)

        move_array = np.array(move_array, dtype=int)
        last_move_array = np.zeros(len(edges_array), dtype=int)

        if agent == 'uav':
            last_edge = self.last_uav_edge
        elif agent == 'ugv':
            last_edge = self.last_ugv_edge
        else:
            last_edge = None

        if last_edge is not None:
            for idx, edge in enumerate(edges_array):
                edge_tuple = tuple(edge)
                if edge_tuple == last_edge or edge_tuple == tuple(reversed(last_edge)):
                    last_move_array[idx] = 1
                    break

        return {
            "edges": edges_array,
            "metadata": metadata_array,
            "traversal_status": traversal_statuses_array,
            "ai_estimates": ai_estimates_array,
            "human_estimates": human_estimates_array,
            "path_array": path_array,
            "move_array": move_array,
            "last_move": last_move_array
        }

    def step_uav(self, action):
        """
        Performs a UAV step based on the given action.
        """
        action = np.asarray(action)

        # Validate the action array
        if len(action) != 3 * self.MAX_NODES:
            raise ValueError(f"Action array must be of size {3 * self.MAX_NODES}")

        # Find the index where the '1' is located
        action_indices = np.where(action == 1)[0]
        if len(action_indices) != 1:
            return -1  # Invalid action

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
        uav_state = self.get_uav_state()
        edges_array = uav_state['edges']

        # Check if the node_action_index is within bounds
        if node_action_index >= len(edges_array):
            raise ValueError("Node action index is out of range")

        # Get the edge corresponding to the action
        edge = edges_array[node_action_index]
        start_node, end_node = edge

        # Determine the destination node
        current_node = self.uav_node
        if start_node == current_node:
            destination_node = end_node
        elif end_node == current_node:
            destination_node = start_node
        else:
            raise ValueError(f"Edge does not connect to current UAV node: {current_node}")

        # Move the UAV in the Mission environment
        backend_edge = self.env.move_uav(destination_node)
        if backend_edge is None:
            return -1  # Invalid move

        self.uav_node = destination_node
        self.last_uav_edge = (start_node, end_node)

        # Perform scanning if required
        if action_type in ['move_scan_ai', 'move_scan_human']:
            # Set the selected edge in the backend
            self.env.selected_edge = backend_edge

            # Get the edge identifier (sorted tuple)
            edge_id = tuple(sorted((start_node, end_node)))

            # Initialize scanning status to current value or 0 if not scanned yet
            current_status = self.uav_scanned_edges.get(edge_id, 0)

            if action_type == 'move_scan_ai':
                self.env.query_ai()
                ai_estimate = self.env.selected_edge.ai_estimate
                self.ai_scanned_edges[edge_id] = ai_estimate

                # Update scanning status
                if current_status == 1:
                    new_status = 3  # Scanned by both
                else:
                    new_status = 2  # Scanned by AI only

                if self.TIMEKEEPER:
                    self.tk.add_tasks("move_uav")
                    self.tk.add_tasks("ai")

            elif action_type == 'move_scan_human':
                self.env.query_human()
                human_estimate = self.env.selected_edge.human_estimate
                self.human_scanned_edges[edge_id] = human_estimate

                # Update scanning status
                if current_status == 2:
                    new_status = 3  # Scanned by both
                else:
                    new_status = 1  # Scanned by human only

                if self.TIMEKEEPER:
                    self.tk.add_tasks("move_uav")
                    self.tk.add_tasks("human")

            # Update the scanning status
            self.uav_scanned_edges[edge_id] = new_status

        else:
            # Just moving, no scanning
            if self.TIMEKEEPER:
                self.tk.add_tasks("move_uav")

    def step_ugv(self, action):
        """
        Performs a UGV step based on the given action.
        """
        action = np.asarray(action)

        # Validate the action array
        if len(action) != self.MAX_NODES:
            raise ValueError(f"Action array must be of size {self.MAX_NODES}")

        # Find the index where the '1' is located
        action_indices = np.where(action == 1)[0]
        if len(action_indices) != 1:
            return -1  # Invalid action

        action_index = action_indices[0]

        # Get the current UGV state
        ugv_state = self.get_ugv_state()
        edges_array = ugv_state['edges']

        # Check if the action_index is within bounds
        if action_index >= len(edges_array):
            raise ValueError("Node action index is out of range")

        # Get the edge corresponding to the action
        edge = edges_array[action_index]
        start_node, end_node = edge

        # Determine the destination node
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
            return -1  # Invalid move
        elif move_result == 0:
            # Landmine detected, UGV returned to original passageway
            self.ugv_traversed_edges[edge_id] = {'status': -1}

            if self.NO_SECOND_TRIES:
                # Attempt to move again to clear the landmine
                move_result = self.env.move_ugv(destination_node)
                if move_result == -1:
                    return -1  # Invalid move
                elif move_result in [1, 2]:
                    # UGV moved successfully
                    self.ugv_node = destination_node
                    self.last_ugv_edge = (start_node, end_node)

                    self.ugv_traversed_edges[edge_id] = {'status': 2}
                    if self.TIMEKEEPER:
                        self.tk.add_tasks("move_ugv")
                        self.tk.add_tasks("clear_mine")
            else:
                if self.TIMEKEEPER:
                    self.tk.add_tasks("move_ugv")
                # Do not attempt to move again
                pass
        elif move_result == 1:
            # Landmine was cleared
            self.ugv_node = destination_node
            self.last_ugv_edge = (start_node, end_node)

            self.ugv_traversed_edges[edge_id] = {'status': 2}
            if self.TIMEKEEPER:
                self.tk.add_tasks("move_ugv")
                self.tk.add_tasks("clear_mine")
        elif move_result == 2:
            # UGV moved successfully without incidents
            self.ugv_node = destination_node
            self.last_ugv_edge = (start_node, end_node)

            if edge_id not in self.ugv_traversed_edges:
                self.ugv_traversed_edges[edge_id] = {'status': 1}
                if self.TIMEKEEPER:
                    self.tk.add_tasks("move_ugv")
        else:
            return -1  # Should not reach here

    def _get_state(self):
        uav_state = self.get_uav_state()
        ugv_state = self.get_ugv_state()

        state = {
            'UAV_STATE': uav_state,
            'UGV_STATE': ugv_state
        }

        return state

    def pretty_state(self):
        """
        Prints the state in a readable format, showing individual UAV and UGV variables.
        """
        state = self.state

        # Extract UAV and UGV states
        uav_state = state['UAV_STATE']
        ugv_state = state['UGV_STATE']

        # UAV variables
        uav_depth = self.uav_depth
        MAX_NODES = self.MAX_NODES

        uav_move_array = uav_state['move_array']
        uav_ai_estimates = uav_state['ai_estimates']
        uav_human_estimates = uav_state['human_estimates']
        uav_metadata = uav_state['metadata']
        uav_traversal = uav_state['traversal_status']
        uav_edges = uav_state['edges']
        uav_path_array = uav_state['path_array']
        uav_last_move = uav_state['last_move']

        # Reshape arrays
        uav_move_array = np.reshape(uav_move_array, (uav_depth, MAX_NODES))
        uav_ai_estimates = np.reshape(uav_ai_estimates, (uav_depth, MAX_NODES))
        uav_human_estimates = np.reshape(uav_human_estimates, (uav_depth, MAX_NODES))
        uav_metadata = np.reshape(uav_metadata, (uav_depth, MAX_NODES))
        uav_traversal = np.reshape(uav_traversal, (uav_depth, MAX_NODES))
        uav_edges = np.reshape(uav_edges, (uav_depth, MAX_NODES, 2))  # Edges are pairs of nodes
        uav_last_move = np.reshape(uav_last_move, (uav_depth, MAX_NODES))

        # Print UAV state
        print("UAV State:")
        for level in range(uav_depth):
            print(f"  Level {level}:")
            print(f"    Move Array       : {uav_move_array[level]}")
            print(f"    Traversal Status : {uav_traversal[level]}")
            print(f"    AI Estimates     : {uav_ai_estimates[level]}")
            print(f"    Human Estimates  : {uav_human_estimates[level]}")
            print(f"    Last Move        : {uav_last_move[level]}")
            print("    Metadata:")
            for i in range(MAX_NODES):
                print(f"      Edge {i}: {uav_metadata[level][i]}")
        print(f"  Path Array: {uav_path_array}")
        print()

        # UGV variables
        ugv_depth = self.ugv_depth

        ugv_move_array = ugv_state['move_array']
        ugv_ai_estimates = ugv_state['ai_estimates']
        ugv_human_estimates = ugv_state['human_estimates']
        ugv_metadata = ugv_state['metadata']
        ugv_traversal = ugv_state['traversal_status']
        ugv_edges = ugv_state['edges']
        ugv_path_array = ugv_state['path_array']
        ugv_last_move = ugv_state['last_move']

        # Reshape arrays
        ugv_move_array = np.reshape(ugv_move_array, (ugv_depth, MAX_NODES))
        ugv_ai_estimates = np.reshape(ugv_ai_estimates, (ugv_depth, MAX_NODES))
        ugv_human_estimates = np.reshape(ugv_human_estimates, (ugv_depth, MAX_NODES))
        ugv_metadata = np.reshape(ugv_metadata, (ugv_depth, MAX_NODES))
        ugv_traversal = np.reshape(ugv_traversal, (ugv_depth, MAX_NODES))
        ugv_edges = np.reshape(ugv_edges, (ugv_depth, MAX_NODES, 2))  # Edges are pairs of nodes
        ugv_last_move = np.reshape(ugv_last_move, (ugv_depth, MAX_NODES))

        # Print UGV state
        print("UGV State:")
        for level in range(ugv_depth):
            print(f"  Level {level}:")
            print(f"    Move Array       : {ugv_move_array[level]}")
            print(f"    Traversal Status : {ugv_traversal[level]}")
            print(f"    AI Estimates     : {ugv_ai_estimates[level]}")
            print(f"    Human Estimates  : {ugv_human_estimates[level]}")
            print(f"    Last Move        : {ugv_last_move[level]}")
            print("    Metadata:")
            for i in range(MAX_NODES):
                print(f"      Edge {i}: {ugv_metadata[level][i]}")
        print(f"  Path Array: {ugv_path_array}")
        print()

    def get_time(self):
        if self.TIMEKEEPER:
            return self.tk.get_status()

        return self.env.total

    def step(self, action):
        """
        Executes the full step in the environment for both UAV and UGV.

        Parameters:
        - action: array
            Indexing:
                [0:MAX_NODES * 1]       = move_uav
                [MAX_NODES * 1: * 2]    = ai
                [MAX_NODES * 2: * 3]    = human
                [MAX_NODES * 3: * 4]    = move_ugv

        Returns:
        - state: current state
        - done: True if the mission is completed, False otherwise
        """
        # Ensure action is a numpy array
        action = np.array(action)
        action_modified = action.copy()

        if self.TIMEKEEPER:
            busy = self.tk.get_next_time()[1]
            task_keys = ['move_uav', 'ai', 'human', 'move_ugv']

            for i, task_key in enumerate(task_keys):
                start_idx = i * self.MAX_NODES
                end_idx = (i + 1) * self.MAX_NODES
                task_action = action[start_idx:end_idx]

                if any(task_action) and busy.get(task_key, 0) == 1:
                    # Conflict detected for this task
                    # Zero out the conflicting action
                    action_modified[start_idx:end_idx] = 0

        # UAV actions
        uav_action = action_modified[:self.MAX_NODES * 3]  # move_uav, ai, human
        if any(uav_action):
            self.step_uav(uav_action)

        # UGV actions
        ugv_action = action_modified[self.MAX_NODES * 3:]  # move_ugv
        if any(ugv_action):
            self.step_ugv(ugv_action)

        if self.TIMEKEEPER:
            self.tk.activate_pending_tasks()

        state = self._get_state()
        done = self.ugv_node == self.GOAL

        return state, done


if __name__ == '__main__':
    env = MissionWrapper("network.json")
