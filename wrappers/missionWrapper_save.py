import networkx as nx  # type: ignore
import numpy as np  # type: ignore
from collections import deque
from base.mission import Mission
from wrappers.timeKeeper import TimeKeeper


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
        self.epoch = 0

    def reset(self, eval=False):
        """
        Resets the environment to its initial state and sets up network and mission properties.

        In evaluation mode (eval=True), the Mission is instantiated with eval=True and uses a fixed seed,
        so that the start and end nodes remain as specified in the configuration.
        In non-evaluation mode, the reset() method randomly selects distinct start and end nodes from the Mission network.
        """
        if eval:
            self.seed = 128
            np.random.seed(self.seed)
        else:
            self.seed = np.random.randint(1000)
        # Instantiate Mission with the appropriate evaluation flag.
        self.env = Mission(self.config_path, eval=eval, seed =self.seed)
        # Call reset() on the Mission to select random nodes if needed.
        self.env.reset()
        
        # Set the goal based on the mission's defined end node.
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

        # Tracking traversal and scanning statuses
        self.ai_scanned_edges = {}
        self.human_scanned_edges = {}
        self.uav_scanned_edges = {}   # To track UAV scanning statuses
        self.ugv_traversed_edges = {}  # To track UGV traversal statuses

        # Reuse the network generated in Mission.
        self.network = self.env.network

        state = self.get_state()
        
        print(self.epoch, eval)
        self.epoch += 1
        
        self.eval_mode = eval
        if self.eval_mode:
            self.eval_step_data = []

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

    def get_map(self):
        return self.network

    def get_uav_state(self):
        """
        Returns the UAV's state up to a specified depth, including additional variables.
        """
        # Get the base UAV state from _get_agent_state
        base_state = self._get_agent_state(self.uav_node, self.uav_depth, agent='uav')

        # Compute the number of connections (edges) between the UAV and UGV
        try:
            num_connections = nx.shortest_path_length(self.network, source=self.uav_node, target=self.ugv_node)
        except nx.NetworkXNoPath:
            num_connections = float('inf')  # No path exists
        # Compute the distance of the UAV from the goal
        try:
            uav_distance_to_goal = nx.shortest_path_length(self.network, source=self.uav_node, target=self.GOAL)
        except nx.NetworkXNoPath:
            uav_distance_to_goal = float('inf')  # No path exists

        # Compute the distance of the UGV from the goal
        try:
            ugv_distance_to_goal = nx.shortest_path_length(self.network, source=self.ugv_node, target=self.GOAL,
                                                           weight='weight')
        except nx.NetworkXNoPath:
            ugv_distance_to_goal = float('inf')  # No path exists

        edges_array = base_state['edges']
        current_node = self.uav_node

        adjacent_nodes = []
        for edge in edges_array:
            start_node, end_node = edge
            if start_node == "None" or end_node == "None":
                adjacent_nodes.append(None)
            else:
                neighbor = end_node if start_node == current_node else start_node
                adjacent_nodes.append(neighbor)

        # Determine the ugv_path_array
        ugv_path_array = np.zeros(len(adjacent_nodes), dtype=int)
        try:
            path_to_ugv = nx.shortest_path(self.network, source=current_node, target=self.ugv_node, weight='weight')
            if len(path_to_ugv) > 1:
                next_node_to_ugv = path_to_ugv[1]
                if next_node_to_ugv in adjacent_nodes:
                    idx = adjacent_nodes.index(next_node_to_ugv)
                    ugv_path_array[idx] = 1
        except nx.NetworkXNoPath:
            pass  # Keep ugv_path_array as zeros if no path exists

        # Add the new variables to the state
        base_state['num_connections'] = num_connections
        base_state['uav_distance_to_goal'] = uav_distance_to_goal
        base_state['ugv_distance_to_goal'] = ugv_distance_to_goal
        base_state['ugv_path_array'] = ugv_path_array

        return base_state

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
            path = nx.shortest_path(self.network, source=current_node, target=self.GOAL, weight='weight')

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
        
        The action array should be of size 2 * MAX_NODES, where:
        - Indices [0, MAX_NODES) correspond to moving and scanning with AI.
        - Indices [MAX_NODES, 2 * MAX_NODES) correspond to moving and scanning with Human.
        
        If more than one value is 1 in the action array, a random index among them will be selected.
        """
        action = np.asarray(action)

        # Validate the action array
        if len(action) != 2 * self.MAX_NODES:
            raise ValueError(f"Action array must be of size {2 * self.MAX_NODES}")

        # Find indices where the action array has a 1
        action_indices = np.where(action == 1)[0]
        if len(action_indices) == 0:
            return -1  # No valid action found

        # If more than one valid action, choose one randomly
        action_index = action_indices[0]

        # Determine the action type based on the index
        if 0 <= action_index < self.MAX_NODES:
            node_action_index = action_index
            action_type = 'move_scan_ai'
        elif self.MAX_NODES <= action_index < 2 * self.MAX_NODES:
            node_action_index = action_index - self.MAX_NODES
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

        # Set the selected edge in the backend and prepare for scanning
        self.env.selected_edge = backend_edge
        edge_id = tuple(sorted((start_node, end_node)))
        current_status = self.uav_scanned_edges.get(edge_id, 0)

        # Perform scanning based on action type
        if action_type == 'move_scan_ai':
            self.env.query_ai()
            ai_estimate = self.env.selected_edge.ai_estimate
            self.ai_scanned_edges[edge_id] = ai_estimate

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

            if current_status == 2:
                new_status = 3  # Scanned by both
            else:
                new_status = 1  # Scanned by human only

            if self.TIMEKEEPER:
                self.tk.add_tasks("move_uav")
                self.tk.add_tasks("human")

        # Update the scanning status for the edge
        self.uav_scanned_edges[edge_id] = new_status


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

    def get_state(self):
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
        state = self.get_state()  # Get the current state

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
        uav_ugv_path_array = uav_state['ugv_path_array']

        # New variables
        num_connections = uav_state['num_connections']
        uav_distance_to_goal = uav_state['uav_distance_to_goal']
        ugv_distance_to_goal = uav_state['ugv_distance_to_goal']

        # Reshape arrays
        uav_move_array = np.reshape(uav_move_array, (uav_depth, MAX_NODES))
        uav_ai_estimates = np.reshape(uav_ai_estimates, (uav_depth, MAX_NODES))
        uav_human_estimates = np.reshape(uav_human_estimates, (uav_depth, MAX_NODES))
        uav_metadata = np.reshape(uav_metadata, (uav_depth, MAX_NODES))
        uav_traversal = np.reshape(uav_traversal, (uav_depth, MAX_NODES))
        uav_edges = np.reshape(uav_edges, (uav_depth, MAX_NODES, 2))  # Edges are pairs of nodes
        uav_last_move = np.reshape(uav_last_move, (uav_depth, MAX_NODES))
        uav_ugv_path_array = np.reshape(uav_ugv_path_array, (uav_depth, MAX_NODES))

        # Print UAV state
        print("UAV State:")
        print(f"  Number of connections between UAV and UGV: {num_connections}")
        print(f"  UAV distance to goal: {uav_distance_to_goal}")
        print(f"  UGV distance to goal: {ugv_distance_to_goal}")
        print(f"  UAV Path Array      : {uav_path_array}")
        for level in range(uav_depth):
            print(f"  Level {level}:")
            print(f"    Move Array       : {uav_move_array[level]}")
            print(f"    UGV Path Array   : {uav_ugv_path_array[level]}")
            print(f"    Traversal Status : {uav_traversal[level]}")
            print(f"    AI Estimates     : {uav_ai_estimates[level]}")
            print(f"    Human Estimates  : {uav_human_estimates[level]}")
            print(f"    Last Move        : {uav_last_move[level]}")
            print("    Metadata:")
            for i in range(MAX_NODES):
                print(f"      Edge {i}: {uav_metadata[level][i]}")
        print()

        # UGV variables (unchanged)
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
        print(f"UGV Path Array : {ugv_path_array}")
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
                [0 : MAX_NODES * 2]       = UAV actions:
                                            [0 : MAX_NODES]       -> move and scan with AI
                                            [MAX_NODES : 2*MAX_NODES] -> move and scan with Human
                [MAX_NODES * 2 : MAX_NODES * 3] = move_ugv

        Returns:
        - state: current state
        - done: True if the mission is completed, False otherwise
        """
        # Ensure action is a numpy array
        action = np.array(action)
        action_modified = action.copy()

        if self.TIMEKEEPER:
            busy = self.tk.get_next_time()[1]
            # Update task keys to match the new action segments:
            # 'ai' for UAV scanning with AI, 'human' for UAV scanning with Human, and 'move_ugv'
            task_keys = ['ai', 'human', 'move_ugv']

            for i, task_key in enumerate(task_keys):
                start_idx = i * self.MAX_NODES
                end_idx = (i + 1) * self.MAX_NODES
                task_action = action[start_idx:end_idx]

                if any(task_action) and busy.get(task_key, 0) == 1:
                    # Conflict detected for this task
                    # Zero out the conflicting action
                    action_modified[start_idx:end_idx] = 0

        # --- Record starting nodes if in evaluation mode ---
        if hasattr(self, 'eval_mode') and self.eval_mode:
            starting_uav_node = self.uav_node  # Record UAV node before move
            starting_ugv_node = self.ugv_node  # Record UGV node before move


        # Process UAV actions (first 2 segments)
        uav_action = action_modified[:self.MAX_NODES * 2]
        if any(uav_action):
            self.step_uav(uav_action)

        # Process UGV actions (last segment)
        ugv_action = action_modified[self.MAX_NODES * 2:]
        if any(ugv_action):
            self.step_ugv(ugv_action)

        if self.TIMEKEEPER:
            self.tk.activate_pending_tasks()

        state = self.get_state()
        done = self.ugv_node == self.GOAL

        # --- If in eval mode, record step info ---
        if hasattr(self, 'eval_mode') and self.eval_mode:
            # Record the ending nodes after move.
            end_uav_node = self.uav_node
            end_ugv_node = self.ugv_node

            record = {
                "action": action,
                "starting_uav_node": starting_uav_node,
                "starting_ugv_node": starting_ugv_node,
                "end_uav_node": end_uav_node,
                "end_ugv_node": end_ugv_node
            }
            self.eval_step_data.append(record)

        return state, done

    def save(self, path):
            """
            Saves the evaluation step data to a CSV file at the given path.
            The evaluation step data includes:
            - step_number
            - action_type (e.g. "move_scan_ai" or "move_scan_human")
            - starting_uav_node
            - starting_ugv_node
            - end_uav_node
            - end_ugv_node
            """
            import pandas as pd
            if hasattr(self, 'eval_step_data'):
                df = pd.DataFrame(self.eval_step_data)
                df.to_csv(path, index=False)
                print(f"Saved evaluation data to {path}")
            else:
                print("No evaluation step data available to save.")


if __name__ == '__main__':
    env = MissionWrapper("network.json")
