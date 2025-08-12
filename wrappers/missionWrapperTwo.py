import networkx as nx
import numpy as np
from base.missionTwo import Mission
from wrappers.timeKeeper import TimeKeeper
import re

class MissionWrapper:
    def __init__(self, max_nodes=3, uav_depth=2, ugv_depth=3,
                 config_path="network_test_two.json", timekeeper=False):
        """
        Initializes the MissionWrapper environment.
        """
        self.config_path = config_path
        self.MAX_NODES   = max_nodes
        self.uav_depth   = uav_depth
        self.ugv_depth   = ugv_depth
        self.TIMEKEEPER  = timekeeper
        self.epoch       = 0

        # These will be set in reset()
        self.uav_node               = None
        self.ugv_node               = None
        self.scanner_uav_node       = None
        self.last_uav_edge          = None
        self.last_ugv_edge          = None
        self.last_scanner_uav_edge  = None
        self.scan_status = {}

    def reset(self, eval=False, test=False):
        """
        Resets the environment and initializes Mission and agent positions.
        """
        # seed and mission
        self.seed = np.random.randint(1000)
        self.env  = Mission(self.config_path, eval=eval, seed=self.seed)
        self.env.reset()

        # start node
        start = self.env.start_node
        self.uav_node         = start
        self.ugv_node         = start
        self.scanner_uav_node = start

        # clear last-edge trackers
        self.last_uav_edge         = None
        self.last_ugv_edge         = None
        self.last_scanner_uav_edge = None

        # initialize scan/traverse records
        self.ai_scanned_edges          = {}
        self.human_scanned_edges       = {}
        self.uav_scanned_edges         = {}
        self.scanner_uav_scanned_edges = {}
        self.scanner_ai_scanned_edges  = {}
        self.ugv_traversed_edges       = {}

        # goal and network
        self.GOAL    = self.env.end_node
        self.network = self.env.network

        # timing for TimeKeeper
        self.TIMES = {
            "human":      self.env._Mission__human_estimate_time,
            "ai":         self.env._Mission__ai_estimate_time,
            "scanner_ai": self.env._Mission__ai_estimate_time,
            "move_uav":   self.env._Mission__uav_traversal_time,
            "move_ugv":   self.env._Mission__ugv_traversal_time,
            "clear_mine": self.env._Mission__ugv_clear_time
        }
        if self.TIMEKEEPER:
            self.tk = TimeKeeper(self.TIMES)

        # build interior nodes for random start/end
        self.nodes = list(self.network.nodes())
        m = int(np.sqrt(len(self.nodes)))
        pattern = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
        self.interior_nodes = []
        for n in self.nodes:
            match = pattern.match(n)
            if match:
                x, y = map(int, match.groups())
                if 5 <= x <= m-5 and 5 <= y <= m-5:
                    self.interior_nodes.append(n)

        state = self.get_state()
        print(self.epoch, eval)
        self.epoch += 1

        # evaluation bookkeeping
        self.eval_mode = eval
        if self.eval_mode:
            self.eval_step_data = []

        return state, {}

    def pad_state(self, edges_array, metadata_array):
        """
        Pads edges and metadata to MAX_NODES.
        """
        dummy_edge = ("None", "None")
        dummy_md   = {"terrain": "unknown", "visibility": 0.0}
        while len(edges_array) < self.MAX_NODES:
            edges_array = np.append(edges_array, [dummy_edge], axis=0)
        while len(metadata_array) < self.MAX_NODES:
            metadata_array = np.append(metadata_array, [dummy_md], axis=0)
        return edges_array, metadata_array

    def get_map(self):
        return self.network

    def get_uav_state(self):
        """
        Returns the UAV's state up to a specified depth, including:
          - local edges, metadata, traversal, estimates, path and move masks
          - num_connections, uav_distance_to_goal, ugv_distance_to_goal
          - ugv_path_array
        """
        base_state = self._get_agent_state(self.uav_node, self.uav_depth, agent='uav')

        # compute num_connections between UAV and UGV
        try:
            num_connections = nx.shortest_path_length(
                self.network,
                source=self.uav_node,
                target=self.ugv_node
            )
        except nx.NetworkXNoPath:
            num_connections = float('inf')

        # compute distances to goal
        try:
            uav_distance_to_goal = nx.shortest_path_length(
                self.network,
                source=self.uav_node,
                target=self.GOAL
            )
        except nx.NetworkXNoPath:
            uav_distance_to_goal = float('inf')

        try:
            ugv_distance_to_goal = nx.shortest_path_length(
                self.network,
                source=self.ugv_node,
                target=self.GOAL,
                weight='weight'
            )
        except nx.NetworkXNoPath:
            ugv_distance_to_goal = float('inf')

        # build ugv_path_array for UAV's neighborhood
        edges_array = base_state['edges']
        adjacent = []
        for start, end in edges_array:
            if start == "None" or end == "None":
                adjacent.append(None)
            else:
                adjacent.append(end if start == self.uav_node else start)

        ugv_path_array = np.zeros(len(adjacent), dtype=int)
        try:
            path_to_ugv = nx.shortest_path(
                self.network,
                source=self.uav_node,
                target=self.ugv_node,
                weight='weight'
            )
            if len(path_to_ugv) > 1:
                next_node = path_to_ugv[1]
                if next_node in adjacent:
                    idx = adjacent.index(next_node)
                    ugv_path_array[idx] = 1
        except nx.NetworkXNoPath:
            pass

        base_state['num_connections']      = num_connections
        base_state['uav_distance_to_goal'] = uav_distance_to_goal
        base_state['ugv_distance_to_goal'] = ugv_distance_to_goal
        base_state['ugv_path_array']       = ugv_path_array

        return base_state

    def get_scanner_uav_state(self):
        """
        Returns the scanner UAV's local edge state up to the same depth,
        plus the same distance metrics and ugv_path_array, so you can
        integrate them if needed.
        """
        base_state = self._get_agent_state(
            self.scanner_uav_node,
            self.uav_depth,
            agent='scanner_uav'
        )

        # same distance computations as for the primary UAV
        try:
            num_connections = nx.shortest_path_length(
                self.network,
                source=self.scanner_uav_node,
                target=self.ugv_node
            )
        except nx.NetworkXNoPath:
            num_connections = float('inf')

        try:
            scanner_distance_to_goal = nx.shortest_path_length(
                self.network,
                source=self.scanner_uav_node,
                target=self.GOAL
            )
        except nx.NetworkXNoPath:
            scanner_distance_to_goal = float('inf')

        # ugv distance unchanged
        try:
            ugv_distance_to_goal = nx.shortest_path_length(
                self.network,
                source=self.ugv_node,
                target=self.GOAL,
                weight='weight'
            )
        except nx.NetworkXNoPath:
            ugv_distance_to_goal = float('inf')

        # build ugv_path_array for scanner
        edges_array = base_state['edges']
        adjacent = []
        for start, end in edges_array:
            if start == "None" or end == "None":
                adjacent.append(None)
            else:
                adjacent.append(end if start == self.scanner_uav_node else start)

        ugv_path_array = np.zeros(len(adjacent), dtype=int)
        try:
            path_to_ugv = nx.shortest_path(
                self.network,
                source=self.scanner_uav_node,
                target=self.ugv_node,
                weight='weight'
            )
            if len(path_to_ugv) > 1:
                next_node = path_to_ugv[1]
                if next_node in adjacent:
                    idx = adjacent.index(next_node)
                    ugv_path_array[idx] = 1
        except nx.NetworkXNoPath:
            pass

        base_state['num_connections']           = num_connections
        base_state['scanner_distance_to_goal']  = scanner_distance_to_goal
        base_state['ugv_distance_to_goal']      = ugv_distance_to_goal
        base_state['ugv_path_array']            = ugv_path_array

        return base_state

    def get_ugv_state(self):
        s = self._get_agent_state(self.ugv_node, self.ugv_depth, agent="ugv")

        # distance along weighted graph to the goal:
        try:
            dist_to_goal = nx.shortest_path_length(
                self.network,
                source=self.ugv_node,
                target=self.GOAL
            )
        except nx.NetworkXNoPath:
            dist_to_goal = float('inf')

        s['ugv_distance_to_goal'] = dist_to_goal
        return s

    def _get_agent_state(self, current_node, depth, agent="uav"):
        """
        Builds the local state for an agent (uav, scanner_uav, or ugv).
        """
        def parse_node(node):
            if isinstance(node, tuple):
                return node
            m = re.match(r'\(\s*([-]?\d+)\s*,\s*([-]?\d+)\s*\)', node)
            if not m:
                raise ValueError(f"Invalid node format: {node}")
            return (int(m.group(1)), int(m.group(2)))

        def format_node(coord):
            return f"({coord[0]}, {coord[1]})"

        # fixed hex directions
        directions = [(-1,0),(0,-1),(1,-1),(1,0),(0,1),(-1,1)]
        MAX_CONN = 6

        edges_level             = []
        metadata_level          = []
        traversal_status_level  = []
        ai_estimates_level      = []
        human_estimates_level   = []

        coord = parse_node(current_node)
        for d in directions:
            nbr = (coord[0] + d[0], coord[1] + d[1])
            if nbr[0] < 0 or nbr[1] < 0:
                # pad
                edges_level.append(("None","None"))
                metadata_level.append({"terrain":"unknown","visibility":0.0})
                traversal_status_level.append(None)
                ai_estimates_level.append(None)
                human_estimates_level.append(None)
                continue

            nbr_str = format_node(nbr)
            if nbr_str in self.network:
                # real edge
                edge_data = self.network.get_edge_data(current_node, nbr_str)
                edges_level.append((current_node, nbr_str))
                metadata_level.append(edge_data)
                eid = tuple(sorted((current_node, nbr_str)))

                # traversal status
                if agent == "ugv":
                    stat = self.ugv_traversed_edges.get(eid, {}).get("status")
                elif agent == "uav":
                    stat = self.uav_scanned_edges.get(eid)
                elif agent == "scanner_uav":
                    stat = self.scanner_uav_scanned_edges.get(eid)
                else:
                    stat = None
                traversal_status_level.append(stat)

                # estimates
                if agent in ("ugv","uav"):
                    ai_estimates_level.append(self.ai_scanned_edges.get(eid))
                    human_estimates_level.append(self.human_scanned_edges.get(eid))
                elif agent == "scanner_uav":
                    ai_estimates_level.append(self.scanner_ai_scanned_edges.get(eid))
                    human_estimates_level.append(None)
                else:
                    ai_estimates_level.append(None)
                    human_estimates_level.append(None)
            else:
                # pad for missing neighbor
                edges_level.append(("None","None"))
                metadata_level.append({"terrain":"unknown","visibility":0.0})
                traversal_status_level.append(None)
                ai_estimates_level.append(None)
                human_estimates_level.append(None)

        # arrays
        edges_array    = np.array(edges_level, dtype=object)
        metadata_array = np.array(metadata_level, dtype=object)
        trav_array     = np.array(traversal_status_level, dtype=object)
        ai_array       = np.array(ai_estimates_level, dtype=object)
        human_array    = np.array(human_estimates_level, dtype=object)

        # shortest-path indicator
        try:
            path = nx.shortest_path(self.network, source=current_node, target=self.GOAL, weight="weight")
            nxt = path[1] if len(path)>1 else None
        except nx.NetworkXNoPath:
            nxt = None
        path_array = np.zeros(MAX_CONN, dtype=int)
        expected = [format_node((coord[0]+d[0],coord[1]+d[1])) for d in directions]
        if nxt in expected:
            path_array[expected.index(nxt)] = 1

        # valid-move mask
        move_array = np.array([1 if e!=("None","None") else 0 for e in edges_level], dtype=int)

        # last move
        last_move = np.zeros(len(edges_level), dtype=int)
        if agent=="uav":
            le = self.last_uav_edge
        elif agent=="ugv":
            le = self.last_ugv_edge
        else:
            le = self.last_scanner_uav_edge
        if le:
            for i,e in enumerate(edges_level):
                if tuple(e)==tuple(le) or tuple(e)==tuple(le[::-1]):
                    last_move[i] = 1
                    break

        # package
        state = {
            "edges":        edges_array,
            "metadata":     metadata_array,
            "traversal":    trav_array,
            "ai_estimates": ai_array,
            "human_estimates":    human_array,
            "path_array":   path_array,
            "move_array":   move_array,
            "last_move":    last_move
        }
        return state

    def step_uav(self, action):
        """
        Move + scan with AI or human for the traversal UAV.
        """
        action = np.asarray(action)
        if len(action) != 2*self.MAX_NODES:
            raise ValueError(f"Action size must be {2*self.MAX_NODES}")
        idxs = np.where(action==1)[0]
        if len(idxs)==0:
            return -1
        ai = idxs[0]
        if ai < self.MAX_NODES:
            node_idx, typ = ai, "ai"
        else:
            node_idx, typ = ai - self.MAX_NODES, "human"

        # get edge
        us = self.get_uav_state()
        edge = us["edges"][node_idx]
        if edge[0]=="None":
            return -1
        s,e = edge
        cur = self.uav_node
        dest = e if s==cur else s

        # move
        be = self.env.move_uav(dest)
        if not be:
            return -1
        self.uav_node        = dest
        self.last_uav_edge   = (s,e)
        self.env.selected_edge = be
        eid = tuple(sorted((s,e)))
        prev = self.uav_scanned_edges.get(eid,0)

        if typ=="ai":
            self.env.query_ai()
            est = be.ai_estimate
            self.ai_scanned_edges[eid] = est
            new = 2 if prev==1 else 2
            if self.TIMEKEEPER:
                self.tk.add_tasks("move_uav")
                self.tk.add_tasks("ai")
        else:
            self.env.query_human()
            est = be.human_estimate
            self.human_scanned_edges[eid] = est
            new = 1 if prev==2 else 1
            if self.TIMEKEEPER:
                self.tk.add_tasks("move_uav")
                self.tk.add_tasks("human")

        self.uav_scanned_edges[eid] = new
        edge_id = tuple(sorted((s, e)))

        # BEFORE you query:
        prev_mask = self.scan_status.get(edge_id, 0)

        if typ == 'ai':
            self.env.query_ai()
            bit = 1
        else:  # 'move_scan_human'
            self.env.query_human()
            bit = 2

        # record the estimate & then update the mask:
        self.scan_status[edge_id] = prev_mask | bit

    def step_scanner_uav(self, action):
        """
        Move + scan with high-reliability AI for the scanner UAV.
        """
        action = np.asarray(action)
        if len(action) != self.MAX_NODES:
            raise ValueError(f"Action size must be {self.MAX_NODES}")
        idxs = np.where(action==1)[0]
        if len(idxs)==0:
            return -1

        node_idx = idxs[0]
        ss = self.get_scanner_uav_state()
        edge = ss["edges"][node_idx]
        if edge[0]=="None":
            return -1
        s,e = edge
        cur = self.scanner_uav_node
        dest = e if s==cur else s

        be = self.env.move_scanner_uav(dest)
        if not be:
            return -1
        self.scanner_uav_node        = dest
        self.last_scanner_uav_edge   = (s,e)
        self.env.selected_edge       = be

        self.env.query_scanner_ai()
        high = be.high_ai_estimate
        eid = tuple(sorted((s,e)))
        self.scanner_ai_scanned_edges[eid]  = high
        self.scanner_uav_scanned_edges[eid] = True

        if self.TIMEKEEPER:
            self.tk.add_tasks("move_uav")
            self.tk.add_tasks("scanner_ai")
        
        edge_id = tuple(sorted((s, e)))
        self.env.query_scanner_ai()  # your new method
        prev_mask = self.scan_status.get(edge_id, 0)
        self.scan_status[edge_id] = prev_mask | 4

    def step_ugv(self, action):
        """
        Move the UGV and handle mines.
        """
        action = np.asarray(action)
        if len(action) != self.MAX_NODES:
            raise ValueError(f"Action size must be {self.MAX_NODES}")
        idxs = np.where(action==1)[0]
        if len(idxs)!=1:
            return -1

        node_idx = idxs[0]
        us = self.get_ugv_state()
        edge = us["edges"][node_idx]
        if edge[0]=="None":
            return -1
        s,e = edge
        cur = self.ugv_node
        dest = e if s==cur else s

        res = self.env.move_ugv(dest)
        eid = tuple(sorted((s,e)))

        if res == -1:
            return -1
        elif res == 0:
            # mine found
            self.ugv_traversed_edges[eid] = {"status": -1}
            # try again
            res2 = self.env.move_ugv(dest)
            if res2 in (1,2):
                self.ugv_node      = dest
                self.last_ugv_edge= (s,e)
                self.ugv_traversed_edges[eid] = {"status": 2}
                if self.TIMEKEEPER:
                    self.tk.add_tasks("move_ugv")
                    self.tk.add_tasks("clear_mine")
        elif res == 1:
            # mine cleared
            self.ugv_node      = dest
            self.last_ugv_edge= (s,e)
            self.ugv_traversed_edges[eid] = {"status": 2}
            if self.TIMEKEEPER:
                self.tk.add_tasks("move_ugv")
                self.tk.add_tasks("clear_mine")
        elif res == 2:
            # moved cleanly
            self.ugv_node      = dest
            self.last_ugv_edge= (s,e)
            if eid not in self.ugv_traversed_edges:
                self.ugv_traversed_edges[eid] = {"status": 1}
            if self.TIMEKEEPER:
                self.tk.add_tasks("move_uav")

    def step(self, action):
        """
        Full environment step: UAV, scanner UAV, then UGV.
        Expects an action vector of length 4*MAX_NODES:
          [0:2M]  -> traversal UAV (AI/human)
          [2M:3M] -> scanner UAV (high-AI)
          [3M:4M] -> UGV
        """
        action = np.asarray(action)
        M = self.MAX_NODES
        if len(action) != 4*M:
            raise ValueError(f"Action must be length {4*M}")

        act = action.copy()
        if self.TIMEKEEPER:
            busy = self.tk.get_next_time()[1]
            keys = ["ai","human","scanner_ai","move_ugv"]
            for i,key in enumerate(keys):
                seg = slice(i*M, (i+1)*M)
                if act[seg].any() and busy.get(key,0)==1:
                    act[seg] = 0

        if self.eval_mode:
            su = self.uav_node
            sg = self.ugv_node

        # apply UAV
        if act[0:2*M].any():
            self.step_uav(act[0:2*M])
        # apply scanner UAV
        if act[2*M:3*M].any():
            self.step_scanner_uav(act[2*M:3*M])
        # apply UGV
        if act[3*M:4*M].any():
            self.step_ugv(act[3*M:4*M])

        if self.TIMEKEEPER:
            self.tk.activate_pending_tasks()

        state = self.get_state()
        done  = (self.ugv_node == self.GOAL)

        if self.eval_mode:
            self.eval_step_data.append({
                "action": action,
                "start_uav": su,
                "start_ugv": sg,
                "end_uav":   self.uav_node,
                "end_ugv":   self.ugv_node
            })

        print(self.eval_mode)
        return state, done

    def get_state(self):
        """
        Returns combined state for traversal UAV, scanner UAV, and UGV.
        """
        return {
            "UAV_STATE":         self.get_uav_state(),
            "SCANNER_UAV_STATE": self.get_scanner_uav_state(),
            "UGV_STATE":         self.get_ugv_state()
        }

    def get_time(self):
        """
        Returns elapsed time or TimeKeeper status.
        """
        if self.TIMEKEEPER:
            return self.tk.get_status()
        return self.env.total

    def save(self, path):
        """
        Save eval data if in eval mode.
        """
        import pandas as pd
        if hasattr(self, "eval_step_data"):
            df = pd.DataFrame(self.eval_step_data)
            df.to_csv(path, index=False)
            print(f"Saved eval data to {path}")
        else:
            print("No eval data to save.")

if __name__ == "__main__":
    env = MissionWrapper("network.json")
    state, _ = env.reset(eval=False)
    print(state)
