import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class MapGenerator:
    def __init__(self, parameters_file, output_json_path=None):
        # Load parameters from JSON file
        with open(parameters_file, 'r') as f:
            params = json.load(f)

        # Load parameters
        self.degree = params['degree']
        self.num_nodes = params['num_nodes']
        self.terrain_types = params['terrain_types']
        self.transition_matrix = np.array(params['transition_matrix'])
        self.precipitation_params = params['precipitation_params']
        self.temp_params = params['temp_params']
        self.wind_params = params['wind_params']
        self.visibility_params = params['visibility_params']
        self.accuracy_params = params['accuracy_params']
        self.mine_likelihood = params['mine_likelihood']
        self.output_json_path = output_json_path
        self.processing_params = params['processing_params']

    def assign_terrain(self, neighbor_terrains):
        num_terrains = len(self.terrain_types)
        prob_distribution = np.zeros(num_terrains)
        for neighbor_terrain in neighbor_terrains:
            idx = self.terrain_types.index(neighbor_terrain)
            prob_distribution += self.transition_matrix[idx]
        prob_distribution /= prob_distribution.sum()
        return np.random.choice(self.terrain_types, p=prob_distribution)

    def generate_precipitation(self, terrain_type):
        params = self.precipitation_params[terrain_type]
        rain_probability = params['rain_probability']
        if np.random.rand() < rain_probability:
            scale = params['scale']
            precipitation = np.random.exponential(scale=scale)
        else:
            precipitation = 0.0
        return np.clip(precipitation, 0, 50)

    def generate_temperature(self, terrain_type, time, precipitation):
        params = self.temp_params[terrain_type]
        base_temp = params['base_temp']
        diurnal_variation = params['diurnal_variation']
        precipitation_effect = params['precipitation_effect']
        temp_variation = diurnal_variation * np.sin(((time - 6) / 24) * 2 * np.pi)
        temperature = base_temp + temp_variation - precipitation_effect * precipitation
        return np.clip(temperature + np.random.normal(0, 2), -10, 45)

    def generate_wind_speed(self, terrain_type, time, precipitation):
        params = self.wind_params[terrain_type]
        base_wind = params['base_wind']
        time_variation = params['time_variation']
        precipitation_effect = params['precipitation_effect']
        wind_variation = time_variation * np.sin((time / 24) * 2 * np.pi)
        wind_speed = base_wind + wind_variation + precipitation_effect * precipitation
        return np.clip(wind_speed + np.random.normal(0, 2), 0, 100)

    def generate_visibility(self, terrain_type, time, precipitation):
        params = self.visibility_params[terrain_type]
        max_visibility = params['max_visibility']
        time_effect = params['time_effect']
        precipitation_effect = params['precipitation_effect']
        visibility = max_visibility
        if time < 7 or time > 19:
            visibility -= time_effect * (abs(13 - time) / 6)
        visibility -= precipitation_effect * precipitation
        return np.clip(visibility + np.random.normal(0, 2), 0, 100)

    def compute_accuracy(self, temperature, wind_speed, visibility,
                         precipitation, terrain_type, noise_std=0.05,
                         visibility_metric=7, visibility_scale=0.55):
        visibility_scaled = visibility / 100.0
        k_vis = visibility_metric
        v0 = visibility_scale
        visibility_effect = 1 / (1 + np.exp(-k_vis * (visibility_scaled - v0)))

        temp_min, temp_max = -10, 45
        temp_scaled = (temperature - temp_min) / (temp_max - temp_min)
        temp_effect = (1 - temp_scaled) * visibility_effect

        wind_speed_max = 100
        wind_speed_scaled = wind_speed / wind_speed_max
        wind_effect = (1 - wind_speed_scaled) * visibility_effect

        precip_max = 50
        precip_scaled = precipitation / precip_max
        precip_effect = np.exp(-2 * precip_scaled) * visibility_effect

        terrain_dict = {'Grassy': 1.0, 'Rocky': 0.5, 'Sandy': 0.0, 'Wooded': -0.25, 'Swampy': -0.75}
        terrain_effect = terrain_dict[terrain_type] * visibility_effect

        intercept, w_vis, w_temp, w_wind, w_precip, w_terrain = -2.0, 1.5, 1.0, 1.0, 1.0, 1.5
        logit_accuracy = (intercept + w_vis * visibility_effect + w_temp * temp_effect + w_wind * wind_effect +
                          w_precip * precip_effect + w_terrain * terrain_effect)
        accuracy = 1 / (1 + np.exp(-logit_accuracy))
        noise = np.random.normal(0, noise_std, size=accuracy.shape)
        return np.clip(accuracy + noise, 0, 1)

    def compute_estimates(self, accuracy, ground_truth, kappa=2, noise_scale=3, threshold=0.5):
        accuracy = np.clip(accuracy, 0, 1)
        GT_sign = 1 if ground_truth else -1
        effective_accuracy = 0 if accuracy <= threshold else (accuracy - threshold) / (1 - threshold)
        mu = kappa * effective_accuracy * GT_sign
        sigma_squared = noise_scale * (1 - effective_accuracy)
        epsilon = np.random.normal(0, np.sqrt(sigma_squared))
        L = mu + epsilon
        return 1 / (1 + np.exp(-L))


    def generate_hexagonal_cell_network(self):
        """
        Generates a hexagonal cell network where each node represents a hexagon.
        Each interior node will have 6 neighbors. Nodes are arranged using axial coordinates.
        Environmental data, mine assignments, and accuracy/estimates are computed as in the original function.
        Also selects start and end nodes from the interior.
        """
        # Determine grid dimensions; here we assume roughly square dimensions.
        m = int(np.sqrt(self.num_nodes))
        n = m
        hex_size = 1  # You can adjust this to scale the layout

        G = nx.Graph()
        pos = {}

        # Generate nodes using axial coordinates (q, r).
        # For a flat grid of hexagons, let q range in [0, m-1] and r in [0, n-1].
        # For pointy-topped hexagons, use the following conversion:
        #   x = hex_size * sqrt(3) * (q + r/2)
        #   y = hex_size * (3/2) * r
        for q in range(m):
            for r in range(n):
                node = (q, r)
                x = hex_size * np.sqrt(3) * (q + r / 2)
                y = hex_size * (3/2) * r
                G.add_node(node, pos=(x, y))
                pos[node] = (x, y)

        # Define axial neighbor offsets for a pointy-topped hex grid.
        # Each interior cell should have 6 neighbors.
        neighbor_directions = [(1, 0), (0, 1), (-1, 1),
                            (-1, 0), (0, -1), (1, -1)]
        for q in range(m):
            for r in range(n):
                node = (q, r)
                for dq, dr in neighbor_directions:
                    neighbor = (q + dq, r + dr)
                    if neighbor in G:
                        G.add_edge(node, neighbor)

        # Set a random current time for environmental data.
        current_time = np.random.randint(0, 24)

        # Assign a random terrain to each node.
        for node in G.nodes():
            G.nodes[node]['terrain'] = np.random.choice(self.terrain_types)

        # For each edge, assign terrain (using one node's terrain), then compute environmental parameters.
        for u, v in G.edges():
            if G.nodes[u].get('terrain'):
                terrain = self.assign_terrain([G.nodes[u]['terrain']])
            elif G.nodes[v].get('terrain'):
                terrain = self.assign_terrain([G.nodes[v]['terrain']])
            else:
                terrain = np.random.choice(self.terrain_types)

            # Update both nodes’ terrain for consistency.
            G.nodes[u]['terrain'] = terrain
            G.nodes[v]['terrain'] = terrain

            precipitation = self.generate_precipitation(terrain)
            temperature   = self.generate_temperature(terrain, current_time, precipitation)
            wind_speed    = self.generate_wind_speed(terrain, current_time, precipitation)
            visibility    = self.generate_visibility(terrain, current_time, precipitation)

            G.edges[u, v].update({
                'terrain': terrain,
                'time': current_time,
                'temperature': temperature,
                'wind_speed': wind_speed,
                'visibility': visibility,
                'precipitation': precipitation,
                'mine_presence': False  # Initialize mine presence to False
            })

        # Assign mines to a random subset of edges.
        num_edges = G.number_of_edges()
        num_mines = int(num_edges * self.mine_likelihood)
        possible_edges = list(G.edges())
        mine_edges_indices = np.random.choice(range(len(possible_edges)), num_mines, replace=False)
        for idx in mine_edges_indices:
            edge = possible_edges[idx]
            G.edges[edge]['mine_presence'] = True

        # Compute accuracy and estimates for each edge.
        for u, v, data in G.edges(data=True):
            human_accuracy = self.compute_accuracy(
                data['temperature'], data['wind_speed'], data['visibility'],
                data['precipitation'], data['terrain'],
                visibility_scale=self.accuracy_params.get('human_accuracy_param', 0.3)
            )
            human_estimate = self.compute_estimates(
                human_accuracy, data['mine_presence'],
                self.accuracy_params.get('kappa', 2), self.accuracy_params.get('noise_scale', 3),
                threshold=self.accuracy_params.get('human_threshold_param', 0)
            )
            ai_accuracy = self.compute_accuracy(
                data['temperature'], data['wind_speed'], data['visibility'],
                data['precipitation'], data['terrain'],
                visibility_scale=self.accuracy_params.get('ai_accuracy_param', 0.3)
            )
            ai_estimate = self.compute_estimates(
                ai_accuracy, data['mine_presence'],
                self.accuracy_params.get('kappa', 2), self.accuracy_params.get('noise_scale', 3),
                threshold=self.accuracy_params.get('ai_threshold_param', 0)
            )
            data.update({
                'human_accuracy': human_accuracy,
                'human_estimate': human_estimate,
                'ai_accuracy': ai_accuracy,
                'ai_estimate': ai_estimate
            })

        # Choose start and end nodes from the interior.
        # For example, select those with q and r at least 2 and at most m-3 and n-3.
        interior_nodes = [node for node in G.nodes() 
                        if node[0] >= 5 and node[0] <= m - 5
                        and node[1] >= 5 and node[1] <= n - 5]
        if interior_nodes:
            start_node = min(interior_nodes, key=lambda x: (x[0], x[1]))
            end_node = max(interior_nodes, key=lambda x: (x[0], x[1]))
        else:
            nodes_list = list(G.nodes())
            start_node = nodes_list[0]
            end_node = nodes_list[-1]

        # Relabel nodes as strings to match your current format and update the positions.
        mapping = {node: str(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        pos = {mapping[node]: p for node, p in pos.items()}
        start_node_str = str(start_node)
        end_node_str = str(end_node)

        return G, pos, start_node_str, end_node_str



    def save_to_json(self, G, start_node, end_node):
        network = {
            "mission": {
                "start": str(start_node),
                "end": str(end_node),
                "human estimate time": self.processing_params.get("human_estimate_time", 30),
                "AI estimate time": self.processing_params.get("ai_estimate_time", 5),
                "UGV traversal time": self.processing_params.get("UGV_traversal_time", 20),
                "UGV clear time": self.processing_params.get("UGV_clear_time", 60),
                "UAV traversal time": self.processing_params.get("UAV_traversal_time", 1)
            },
            "nodes": [
                {
                    "id": str(node)
                } for node in G.nodes()
            ],
            "edges": [
                {
                    "from": str(u),
                    "to": str(v),
                    "landmine_present": data["mine_presence"],
                    "landmine_cleared": False,
                    "metadata": {
                        "terrain": data["terrain"],
                        "time": data["time"],
                        "temperature": data["temperature"],
                        "wind_speed": data["wind_speed"],
                        "visibility": data["visibility"],
                        "precipitation": data["precipitation"],
                        "weight": 50
                    },
                    "human_estimate": data.get("human_estimate", 0),
                    "ai_estimate": data.get("ai_estimate", 0),
                    "ai_queried": False,
                    "human_queried": False,
                    "inaccessible": {
                        "mine_presence": data["mine_presence"],
                        "human_accuracy": data.get("human_accuracy", 0),
                        "ai_accuracy": data.get("ai_accuracy", 0)
                    }
                } for u, v, data in G.edges(data=True)
            ]
        }
        with open(self.output_json_path, 'w') as f:
            json.dump(network, f, indent=4)

    def visualize_network(self, G, pos):
        terrain_colors = {
            'Grassy': 'green',
            'Rocky': 'gray',
            'Sandy': 'yellow',
            'Wooded': 'darkgreen',
            'Swampy': 'brown'
        }
        edge_colors = []
        for u, v in G.edges():
            # If the edge has a mine, color it black.
            if G.edges[u, v].get('mine_presence', False):
                edge_colors.append('black')
            else:
                terrain = G.edges[u, v]['terrain']
                edge_colors.append(terrain_colors.get(terrain, 'gray'))
        
        plt.figure(figsize=(8, 6))
        node_sizes = [10 for _ in G.nodes()]  # Smaller nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=10)
        #nx.draw_networkx_labels(G, pos, font_size=6, font_color='black')
        plt.title("Hexagonal Lattice Network with Terrain and Mines")
        plt.axis("off")
        
        # Create legend handles for each terrain type and mine
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=color, label=label) for label, color in terrain_colors.items()]
        mine_patch = mpatches.Patch(color='black', label='Mine')
        patches.append(mine_patch)
        plt.legend(handles=patches, loc='upper right')
        
        plt.show()



if __name__ == '__main__':
    parameters_file = "config/testing_parameters.json"  # Path to JSON with parameters
    output_json_path = "config/network_test.json"  # Path to save the generated network
    map_generator = MapGenerator(parameters_file, output_json_path)
    
    # Generate a hexagonal (planar) network instead of a random regular graph
    G, pos, start_node, end_node = map_generator.generate_hexagonal_cell_network()
    
    # Optionally save the network to JSON (you might need to choose start/end nodes differently)
    # For example, using the first and last node in G:

    map_generator.save_to_json(G, start_node, end_node)
    
    # Visualize the hexagonal grid
    map_generator.visualize_network(G, pos)
