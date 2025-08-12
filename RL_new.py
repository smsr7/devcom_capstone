from wrappers.missionWrapper import MissionWrapper
from prediction.predictor import LinearRegressor
from mapGenerator import MapGenerator
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import gymnasium as gym  # type: ignore
import datetime
import networkx as nx

import torch
device = torch.device("cpu")

class Environment(MissionWrapper, gym.Env):

    def __init__(self, save_dir=None):
        uav_depth = 1
        ugv_depth = 1

        MAX_NODES = 6

        self.BASE_WEIGHT = 50

        self.aiRegressor = LinearRegressor(target_variable='ai_pred')
        self.humanRegressor = LinearRegressor(target_variable='huma_pred')

        self.aiRegressor.load_model('prediction/ai_regressor_model.pkl')
        self.humanRegressor.load_model('prediction/huma_regressor_model.pkl')

        super().__init__(uav_depth=uav_depth, ugv_depth=ugv_depth, max_nodes=MAX_NODES, timekeeper=False)

        # Action space: Binary actions for UAV and UGV across nodes
        self.action_space = gym.spaces.MultiBinary(self.MAX_NODES * 2)

        # Ovservation space: This was determined by running RLagent.py
        observation_shape = (68,)
        # print(f"Original observation space shape: {observation_shape}")

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,  # Example; adjust based on `_process_uav_state`
            dtype=np.float32,
        )
        # Internal variables
        self.state = None

        self.metrics_list = []
        self.run_counter = 0

        self.save_dir = save_dir

    def make_prediction(self, metadata):
        # Convert metadata to DataFrame
        df = pd.json_normalize(metadata)

        numerical_vars = ['temperature', 'wind_speed', 'visibility', 'precipitation']
        df[numerical_vars] = df[numerical_vars].fillna(0)

        if 'mines' not in df.columns:
            df['mines'] = 1

        output_metadata = df.drop(columns=['mines'], errors='ignore')
        model_features = ['temperature', 'wind_speed', 'visibility', 'precipitation', 'mines']

        df_no_mine = df.copy()
        df_no_mine['mines'] = 1
        df_mine = df.copy()
        df_mine['mines'] = 2

        df_no_mine = df_no_mine[model_features]
        df_mine = df_mine[model_features]

        # Make predictions for no mine present
        prediction_ai_no_mine = self.aiRegressor.predict(df_no_mine)
        prediction_human_no_mine = self.humanRegressor.predict(df_no_mine)

        # Make predictions for mine present
        prediction_ai_mine = self.aiRegressor.predict(df_mine)
        prediction_human_mine = self.humanRegressor.predict(df_mine)

        # Compile results
        results = []
        for i in range(len(df)):
            result = {
                'ai_pred_no_mine': prediction_ai_no_mine[i],
                'human_pred_no_mine': prediction_human_no_mine[i],
                'ai_pred_mine': prediction_ai_mine[i],
                'human_pred_mine': prediction_human_mine[i]
            }
            results.append(result)

        out = pd.concat([output_metadata, pd.DataFrame(results)], axis=1)
        prediction_columns = ['ai_pred_no_mine', 'human_pred_no_mine', 'ai_pred_mine', 'human_pred_mine']
        out.loc[out['terrain'] == 'unknown', prediction_columns] = 0

        return out


    def onehot_to_continuous_embedding(self, onehot_array):
        """
        Converts a one-hot vector (length 6) into a 2D continuous embedding
        using cosine and sine of an angle corresponding to the index.
        """
        num_directions = len(onehot_array)
        # Find index in the one-hot array (assumes one element is 1)
        index = np.argmax(onehot_array)
        # Calculate the angle (assume equally spaced directions around the circle)
        angle = index * (2 * np.pi / num_directions)
        # Return a 2D vector (cosine and sine of the angle)
        return np.array([np.cos(angle), np.sin(angle)])

    def _process_uav_state(self, state):
        # Generate predictions from metadata
        metadata_state = self.make_prediction(state['UAV_STATE']['metadata'])
        def replace_none_with_zero(array):
            return np.array([0 if x is None else x for x in array])

        traversal_status = replace_none_with_zero(state['UAV_STATE']['traversal_status'])
        ai_estimates = replace_none_with_zero(state['UAV_STATE']['ai_estimates'])
        human_estimates = replace_none_with_zero(state['UAV_STATE']['human_estimates'])

        # Store additional variables for later use
        self.processed_ai_estimates = ai_estimates
        self.processed_human_estimates = human_estimates
        self.ai_confidence = np.abs(ai_estimates - 0.5)  # higher value = more confident (i.e. away from 0.5)
        self.human_confidence = np.abs(human_estimates - 0.5)

        path_array = np.array(state['UAV_STATE']['path_array'])
        move_array = np.array(state['UAV_STATE']['move_array'])
        last_move = np.array(state['UAV_STATE']['last_move'])
        ugv_path_array = np.array(state['UAV_STATE']['ugv_path_array'])

        # Drop unnecessary columns from metadata (time and terrain)
        metadata_state_no_time = metadata_state.drop(columns=['time', 'terrain'])
        metadata_values = metadata_state_no_time.values.flatten()

        # ['edges', 'metadata', 'traversal_status', 'ai_estimates', 'human_estimates', 
        # 'path_array', 'move_array', 'last_move', 'num_connections', 'uav_distance_to_goal', 
        # 'ugv_distance_to_goal', 'ugv_path_array']
        # Concatenate all parts into a single state vector
        state_vector = np.concatenate([
            traversal_status,
            self.onehot_to_continuous_embedding(path_array),
            move_array,
            self.onehot_to_continuous_embedding(last_move),
            [state['UAV_STATE']['num_connections'],
            state['UAV_STATE']['uav_distance_to_goal'],
            state['UAV_STATE']['ugv_distance_to_goal']
            ],
            self.onehot_to_continuous_embedding(ugv_path_array)
        ])

        return state_vector

    def update_edge_weight(self, edge, weight):
        if edge is None:
            print("Warning: Attempted to update edge weight, but the edge is None.")
            return  # Safely exit; only relevant on first edge where it cannot access "previous edge"
        else:
            u, v = edge
            if self.network.has_edge(u, v):
                self.network[u][v]['weight'] = weight
            else:
                print(f"Edge ({u}, {v}) not found in the network.")

    def reset(self, delay_start=20, max_iter=400, max_scans=300, regen=False, eval=False, test=False, **kwargs):
        """
        Resets the environment to its initial state and sets up network and mission properties.
        
        If eval=True, the underlying Mission is reset in evaluation mode (with fixed start/end nodes).
        """
        # Pass the eval flag to the parent reset method (which instantiates the Mission accordingly)
        state = super().reset(eval=eval)

        self.cumulative_ugv_weights = None

        self.max_scans = max_scans
        self.test = test
    
        self.uav_goal = False
        if regen:
            parameters_file = "config/testing_parameters.json"  # Path to JSON with parameters
            output_json_path = "config/network.json"              # Path to save the generated network
            map_generator = MapGenerator(parameters_file, output_json_path)
            G, start_node, end_node = map_generator.generate_network()

        self.state = state
        self.previous_ugv_distance = self.state['UAV_STATE']['ugv_distance_to_goal']

        state = self._process_uav_state(self.state)
        self.last_state = np.concatenate([state.copy(), np.zeros(13)])

        obs = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.array(obs, dtype=np.float32).flatten()

        # --- Padding: Ensure observation has the expected dimension ---
        expected_dim = self.observation_space.shape[0]  # e.g., 201
        if obs.shape[0] < expected_dim:
            padding = np.zeros(expected_dim - obs.shape[0], dtype=np.float32)
            obs = np.concatenate([obs, padding])
        elif obs.shape[0] > expected_dim:
            obs = obs[:expected_dim]
        # ---------------------------------------------------------------
        
        self.iter = 0
        self.max_iter = max_iter
        self.delay_start = delay_start
        self.move_ugv = False

        # Initialize metrics for the current episode
        self.current_episode_metrics = {
            'run': self.run_counter,
            'actions': [],
            'iterations': 0,
            'ugv_start': 0,
            'mines_encountered': 0,
            'total_time': 0,
            'reward': 0,
            'path_confidence': 0
        }
        self.old_distance = 0

        return obs, None

    def step(self, action):
        done = False
        # --- Limit action to a single 1 ---
        action = np.array(action)
        out_action = action.copy()

        # Get the UAV state's move array
        uav_move = self.state['UAV_STATE']['move_array']
        uav_move = np.append(uav_move, uav_move.copy())

        # Restrict indices to those where both action and uav_move are 1
        ones = np.where((action == 1) & (uav_move == 1))[0]
        err_reward = 0
        if len(ones) > 1:
            chosen_index = ones[0]  
            err_reward = -15000
        elif len(ones) == 1:
            chosen_index = ones[0]
        else:
            chosen_index = np.random.choice(np.where(uav_move == 1)[0])
            err_reward = -50000

        new_action = np.zeros_like(action)
        new_action[chosen_index] = 1
        action = new_action
        # Save the filtered action as the previous action
        self.previous_action = action.copy()
        
        # --- Append UGV action segment if needed ---
        if len(action) < self.MAX_NODES * 3:
            ugv_action = np.zeros(self.MAX_NODES)
            if self.move_ugv:
                ugv_action[np.where(self.state['UGV_STATE']['path_array'] == 1)[0]] = 1
                action = np.concatenate((action, ugv_action))
        
        if self.state['UAV_STATE']['traversal_status'][chosen_index % self.MAX_NODES] is not None:
            dupe_reward = True
        else:
            dupe_reward = False

        if chosen_index == -1:
            dupe_reward = True

        
        state, done = super().step(action)
        
        # Process UAV edge scanning results to update edge weights
        if self.last_uav_edge is not None:
            # Identify the index of the last move from the UAV state.
            move_indices = np.where(state['UAV_STATE']['last_move'] != 0)[0]
            if len(move_indices) > 0:
                idx = move_indices[0]
                human_estimate = state['UAV_STATE']['human_estimates'][idx]
                ai_estimate = state['UAV_STATE']['ai_estimates'][idx]
        
                if human_estimate is not None:
                    self.update_edge_weight(self.last_uav_edge, 100 * human_estimate)
                elif ai_estimate is not None:
                    self.update_edge_weight(self.last_uav_edge, 100 * ai_estimate)
        
        # Update the state after performing the step.
        state = self.get_state()
        self.state = state
        state = self._process_uav_state(self.state)
        
        obs = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        # --- Append previous action to the state ---
        obs = np.concatenate([obs, self.previous_action])
        obs = np.concatenate([obs, np.asarray([self.iter])])

        state = np.concatenate([obs.copy(), self.last_state])
        
        self.last_state = obs.copy()
        
        # --- Compute scan confidence (as before) ---
        scan_confidence = 0
        last_move_indices = np.where(self.state['UAV_STATE']['last_move'] == 1)[0]
        if len(last_move_indices) > 0:
            idx = last_move_indices[0]
            ai_pred = self.state['UAV_STATE']['ai_estimates'][idx]
            human_pred = self.state['UAV_STATE']['human_estimates'][idx]
            predictions = []
            if ai_pred is not None:
                predictions.append(ai_pred)
            if human_pred is not None:
                predictions.append(human_pred)
            if predictions:
                scan_confidence = np.mean(np.abs(np.array(predictions) - 0.5))
        self.last_scan_confidence = scan_confidence
        # -------------------------------------
        
        # Compute reward and determine if mission is completed
        done = self.state['UAV_STATE']['ugv_distance_to_goal'] == 0
    
        if self.iter > self.max_iter:
            done = True

        if self.last_ugv_edge is not None:
            last_weight = self.network[self.last_ugv_edge[0]][self.last_ugv_edge[1]]['weight']
            if hasattr(self, 'cumulative_ugv_weights') and self.cumulative_ugv_weights is not None:
                self.cumulative_ugv_weights = np.append(self.cumulative_ugv_weights, last_weight)
            else:
                self.cumulative_ugv_weights = np.array([last_weight])

        try:
            path = nx.shortest_path(self.network, source=self.ugv_node, target=self.GOAL, weight='weight')
        except nx.NetworkXNoPath:
            path = []

        # Compute weights along the remaining path.
        if len(path) > 1:
            remaining_weights = np.array([self.network[u][v]['weight'] for u, v in zip(path[:-1], path[1:])])
        else:
            remaining_weights = np.array([])

        # If the UGV has already moved (self.iter > self.delay_start) and cumulative UGV path weights are tracked,
        # combine them with the remaining path weights.
        if self.move_ugv and hasattr(self, 'cumulative_ugv_weights'):
            all_weights = np.concatenate([self.cumulative_ugv_weights, remaining_weights])
        else:
            all_weights = remaining_weights

        # Compute the average weight; if no weights are available, default to 50.
        if all_weights.size > 0:
            self.avg_weight = all_weights.mean()
        else:
            self.avg_weight = 50

        if self.avg_weight < 35 or self.iter > self.max_scans or (self.test and self.iter > self.delay_start):
            if not self.move_ugv:
                self.current_episode_metrics['ugv_start'] = self.iter
            self.move_ugv = True
            
            

        reward = self.calculate_reward(self.state, out_action, done, dupe_reward)
        reward += err_reward
        self.iter += 1
        
        # Collect metrics
        self.current_episode_metrics['actions'] = action
        self.current_episode_metrics['iterations'] = self.iter
        
        traversal_status = self.state['UGV_STATE']['traversal_status']
        last_move_idx = np.where(self.state['UGV_STATE']['last_move'] == 1)[0]
        if len(last_move_idx) > 0:
            mines_encountered = 1 if traversal_status[last_move_idx[0]] == 2 else 0
            self.current_episode_metrics['mines_encountered'] += mines_encountered
        
        total_time = self.get_time()
        self.current_episode_metrics['total_time'] = total_time
        self.current_episode_metrics['reward'] += reward
        
        if done:
            self.metrics_list.append(self.current_episode_metrics)
            self.run_counter += 1  # Increment run counter for the next episode
        
        info = {}
        
        
        return state, reward, done, None, info

    def get_metrics_dataframe(self):
        # Convert the list of metrics into a pandas DataFrame
        df = pd.DataFrame(self.metrics_list)
        df.set_index('run', inplace=True)
        self.metrics_list = []
        return df

    def render(self, mode='human'):
        """
        Render the current environment state.
        """
        print(f"State: {self.state}")

    def calculate_reward(self, state, action, done, dupe_reward=False):
        """
        Compute the reward for the current state and action.
        
        Reward components include:
        - Distance improvement (scaled by dist_mult)
        - An action penalty/bonus
        - A radius reward based on the scan counts
        - A GOAL_BONUS when the UAV reaches 0 distance
        - A new path confidence component based on the UGV path:
                * The shortest path from self.ugv_node to self.GOAL is computed using the network edge weights.
                * If self.iter > self.delay_start and cumulative UGV path weights (self.cumulative_ugv_weights) are tracked,
                they are combined with the remaining path weights.
                * The average weight is then used to compute a bonus/penalty:
                    - If avg_weight > 30: negative reward proportional to (avg_weight - 30)
                    - If 20 <= avg_weight <= 30: an exponentially increasing bonus as the average approaches 20
                    - If avg_weight < 20: a large bonus.
        """
        # --- Existing Reward Components ---
        dist_mult = 0
        path_confidence_weight = 10
        radius_mult = 2
        bomb_mult = 0
        action_mult = 5
        uav_mult = 10
        confidence_mult = 0

        dupe_reward_val = 1

        reward = 0

        
        ugv_distance = state['UAV_STATE']['ugv_distance_to_goal']
        
        previous_ugv_distance = self.previous_ugv_distance

        # Compute distance improvement reward. (Note: using abs() here; adjust if you want directional improvement.)
        distance_reward = (abs(previous_ugv_distance - ugv_distance)
                        if self.move_ugv and not done
                        else abs((previous_ugv_distance - ugv_distance) - self.BASE_WEIGHT))

        # Action reward: bonus if exactly one element is active; otherwise, a penalty.
        action_reward = -1 * (action.sum() - 1) if action.sum() != 1 else 100

        # Radius reward based on scan counts (combining both AI and human estimates).
        ai_count = np.count_nonzero(state['UGV_STATE']['ai_estimates'] is not None)
        human_count = np.count_nonzero(state['UGV_STATE']['human_estimates'] is not None)
        radius_reward = (ai_count + human_count)

        # Bomb penalty (currently set to zero).
        traversal_status = state['UGV_STATE']['traversal_status']
        last_move_idx = np.where(state['UGV_STATE']['last_move'] == 1)[0]
        bomb_reward = 0  # (adjust if needed)

        # Update previous UGV distance for next step.
        self.previous_ugv_distance = ugv_distance

        confidence_reward = self.last_scan_confidence * 100
        # Combine the basic reward components.
        reward = (dist_mult * distance_reward +
                radius_mult * radius_reward +
                bomb_mult * bomb_reward +
                action_mult * action_reward + 
                confidence_mult * confidence_reward)

        
        reward -= self.iter * 10 if self.iter > 200 else 0

        reward += dupe_reward_val if dupe_reward else 0

        # --- Goal Bonus ---
        uav_distance = state['UAV_STATE']['uav_distance_to_goal']
        GOAL_BONUS = 10000

        # Check for goal reached.
        if uav_distance == 0:
            print("GOAL_REACHED")
            if not self.uav_goal:
                reward += GOAL_BONUS
            self.uav_goal = True

        # If the goal hasn't been reached yet, give a reward only when the UAV sets a new record (lowest distance)
        if not self.uav_goal:
            # Initialize best_distance if not already done
            if not hasattr(self, 'best_distance'):
                # You could initialize to a very high value (or you can set it during the episode initialization)
                self.best_distance = state['UAV_STATE']['uav_distance_to_goal']
            
            # Only reward if the UAV's current distance is lower than any previously recorded distance.
            if uav_distance < self.best_distance:
                uav_reward = 100 * (self.best_distance - uav_distance)
                reward += uav_reward * uav_mult
                self.best_distance = uav_distance

        # Optionally, you can still update the old_distance if needed for other purposes.
        self.old_distance = uav_distance
        # --- Path Confidence Component (UGV Path Based) ---
        # Compute the current shortest path from the UGV node to the GOAL.
        

        # Compute the path confidence reward based on the average weight:
        # - If avg_weight > 30: negative reward proportional to (avg_weight - 30)
        # - If 20 <= avg_weight <= 30: exponentially increasing reward as avg_weight approaches 20
        # - If avg_weight < 20: a large bonus.
        path_confidence_reward = 100 * (35 - self.avg_weight)
        if self.avg_weight < 25:
            reward += 25000

        reward += path_confidence_reward * path_confidence_weight
        if done:
            self.current_episode_metrics['path_confidence'] = self.avg_weight

        return reward

def train(timestamp = None):
    """
    Trains a PPO agent with periodic evaluation.
    Before training begins, an initial evaluation episode is run (with eval=True)
    and its metrics are logged to CSV. Then training continues with periodic evaluation
    episodes. A folder with the timestamp is created to store results, and whenever
    the last evaluation reports 0 mines encountered, the model is saved into a 'saves'
    subfolder.
    """
    import datetime
    import os
    import pandas as pd
    from stable_baselines3 import PPO  # type: ignore

    # Create timestamp and directories for saving results and models
    now = datetime.datetime.now()
    if timestamp is None:
        timestamp = now.strftime('%Y-%m-%d_%H-%M')
    
    results_folder = os.path.join("results", timestamp)
    os.makedirs(results_folder, exist_ok=True)
    saves_folder = os.path.join(results_folder, "saves")
    os.makedirs(saves_folder, exist_ok=True)

    # Initialize the environment (training mode by default)
    env = Environment(save_dir=timestamp)
    
    # Initialize the PPO model
    model = PPO('MlpPolicy', env, verbose=1)
    
    eval_metrics_list = []
    
    # --- Initial Evaluation Episode (Before Training) ---
    print("Running initial evaluation episode before training...")
    obs, _ = env.reset(eval=True)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
    
    initial_metrics_df = env.get_metrics_dataframe()
    print(initial_metrics_df['mines_encountered'].values[0])
    if initial_metrics_df['mines_encountered'].values[0] < 6:
        model = None
        train(timestamp=timestamp)
        return
    env.save(os.path.join(saves_folder, f"out_{0}.csv"))
    initial_metrics_df["eval"] = True
    eval_metrics_list.append(initial_metrics_df)
    initial_eval_csv = os.path.join(results_folder, f"eval_metrics_{timestamp}.csv")
    initial_metrics_df.to_csv(initial_eval_csv, index=False)
    print("Initial evaluation complete.")

    total_timesteps = 50000000
    eval_interval = 1250  # number of timesteps per training chunk
    timesteps_trained = 0

    while timesteps_trained < total_timesteps:
        # Train for a chunk of timesteps
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        timesteps_trained += env.epoch  # env.epoch should reflect timesteps learned

        # --- Periodic Evaluation Phase ---
        obs, _ = env.reset(eval=True)
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
        
        env.save(os.path.join(saves_folder, f"out_{env.epoch}.csv"))
        metrics_df = env.get_metrics_dataframe()
        # Mark all rows as non-eval except the last row
        metrics_df["eval"] = False
        if not metrics_df.empty:
            metrics_df.loc[metrics_df.index[-1], "eval"] = True
        eval_metrics_list.append(metrics_df)
        
        # Save model if no mines were encountered in the last evaluation
        if metrics_df.loc[metrics_df.index[-1], "mines_encountered"] <= 8:
            model_save_path = os.path.join(saves_folder, f"{env.epoch}.zip")
            model.save(model_save_path)

        # Concatenate all evaluation metrics and save CSV
        all_eval_metrics = pd.concat(eval_metrics_list, ignore_index=True)
        all_eval_csv = os.path.join(results_folder, f"eval_metrics_{timestamp}.csv")
        all_eval_metrics.to_csv(all_eval_csv, index=False)
        print(f"Completed evaluation at {timesteps_trained} timesteps.")

    # --- Final Test Episode (in training mode) ---
    obs, _ = env.reset(eval=False)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
    final_metrics_df = env.get_metrics_dataframe()
    final_csv = os.path.join(results_folder, f"metrics_{timestamp}.csv")
    final_metrics_df.to_csv(final_csv, index=False)
   
    print("Final test info:", info)

def train2(timestamp = None):
    """
    Trains a PPO agent with periodic evaluation.
    Before training begins, an initial evaluation episode is run (with eval=True)
    and its metrics are logged to CSV. Then training continues with periodic evaluation
    episodes. A folder with the timestamp is created to store results, and whenever
    the last evaluation reports 0 mines encountered, the model is saved into a 'saves'
    subfolder.
    """
    import datetime
    import os
    import pandas as pd
    from stable_baselines3 import PPO  # type: ignore

    # Create timestamp and directories for saving results and models
    now = datetime.datetime.now()
    if timestamp is None:
        timestamp = now.strftime('%Y-%m-%d_%H-%M')
    
    results_folder = os.path.join("results", timestamp)
    os.makedirs(results_folder, exist_ok=True)
    saves_folder = os.path.join(results_folder, "saves")
    os.makedirs(saves_folder, exist_ok=True)

    # Initialize the environment (training mode by default)
    env = Environment(save_dir=timestamp)
    
    # Initialize the PPO model
    model = PPO.load("results/2025-04-13_19-09/saves/1303.zip", env=env)
    
    eval_metrics_list = []
    
    # --- Initial Evaluation Episode (Before Training) ---
    print("Running initial evaluation episode before training...")
    obs, _ = env.reset(eval=True)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
    
    initial_metrics_df = env.get_metrics_dataframe()
    print(initial_metrics_df['mines_encountered'].values[0])
    '''if initial_metrics_df['mines_encountered'].values[0] <= 7:
        model = None
        train(timestamp=timestamp)
        return'''
    env.save(os.path.join(saves_folder, f"out_{0}.csv"))
    initial_metrics_df["eval"] = True
    eval_metrics_list.append(initial_metrics_df)
    initial_eval_csv = os.path.join(results_folder, f"eval_metrics_{timestamp}.csv")
    initial_metrics_df.to_csv(initial_eval_csv, index=False)
    print("Initial evaluation complete.")

    total_timesteps = 50000000
    eval_interval = 500  # number of timesteps per training chunk
    timesteps_trained = 0

    while timesteps_trained < total_timesteps:
        # Train for a chunk of timesteps
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        timesteps_trained += env.epoch  # env.epoch should reflect timesteps learned

        # --- Periodic Evaluation Phase ---
        obs, _ = env.reset(eval=True)
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
        
        env.save(os.path.join(saves_folder, f"out_{env.epoch}.csv"))
        metrics_df = env.get_metrics_dataframe()
        # Mark all rows as non-eval except the last row
        metrics_df["eval"] = False
        if not metrics_df.empty:
            metrics_df.loc[metrics_df.index[-1], "eval"] = True
        eval_metrics_list.append(metrics_df)
        
        # Save model if no mines were encountered in the last evaluation
        if metrics_df.loc[metrics_df.index[-1], "mines_encountered"] <= 9:
            model_save_path = os.path.join(saves_folder, f"{env.epoch}.zip")
            model.save(model_save_path)

        # Concatenate all evaluation metrics and save CSV
        all_eval_metrics = pd.concat(eval_metrics_list, ignore_index=True)
        all_eval_csv = os.path.join(results_folder, f"eval_metrics_{timestamp}.csv")
        all_eval_metrics.to_csv(all_eval_csv, index=False)
        print(f"Completed evaluation at {timesteps_trained} timesteps.")

    # --- Final Test Episode (in training mode) ---
    obs, _ = env.reset(eval=False)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
    final_metrics_df = env.get_metrics_dataframe()
    final_csv = os.path.join(results_folder, f"metrics_{timestamp}.csv")
    final_metrics_df.to_csv(final_csv, index=False)
   
    print("Final test info:", info)


def main(maxSteps=1000):
    #  code to go straight to goal
    import os
    #  code to go straight to goal
    save_dir = 'results/results/baselines'
    # Initialize the environment (training mode by default)
    eval_metrics_list = []

    for i in range(1):
        env = Environment()
        env.reset(delay_start=10, eval=True, test=True)
        env.iter = 0
        done = False
        total_mines = 0

        while not done:
            action = np.zeros(env.MAX_NODES * 2)
            action[np.where(env.state['UAV_STATE']['path_array'] == 1)[0] + 6] = 1
            state, reward, done, _, info = env.step(action)
            traversal_status = env.state['UGV_STATE']['traversal_status']
            last_move_idx = np.where(env.state['UGV_STATE']['last_move'] == 1)[0]
            
            if len(last_move_idx) > 0:
                mines_encountered = 1 if traversal_status[last_move_idx[0]] == 2 else 0
                total_mines += mines_encountered
        
        initial_metrics_df = env.get_metrics_dataframe()
        print(initial_metrics_df['mines_encountered'].values[0])

        env.save(os.path.join(save_dir, f"out_human_single_{i}.csv"))
        initial_metrics_df["eval"] = True
        eval_metrics_list.append(initial_metrics_df)

    metrics_df = pd.concat(eval_metrics_list)
    initial_eval_csv = os.path.join(save_dir, f"eval_metrics.csv")
    metrics_df.to_csv(initial_eval_csv, index=False)
       


def main2(maxSteps=1000):
    import os
    #  code to go straight to goal
    save_dir = 'results/results/baselines'
    # Initialize the environment (training mode by default)
    eval_metrics_list = []
    for i in range(1):
        
        env = Environment(save_dir=save_dir)
        
        env.reset(delay_start=80, eval=True, test=True)
        env.iter = 0
        done = False
        total_mines = 0
        uav_goal = False

        while not done:
            action = np.zeros(env.MAX_NODES * 2)
            
            uav_dist = env.state['UAV_STATE']['uav_distance_to_goal']

            if uav_dist == 0:
                uav_goal = True
        
            if not uav_goal:
                action[np.where(env.state['UAV_STATE']['path_array'] == 1)[0] + 6] = 1
            else:
                ugv_path = np.where(env.state['UAV_STATE']['ugv_path_array'] == 1)[0]
                
                if len(ugv_path) == 0:
                    print("No UGV path found.")
                else:
                    ugv_index = ugv_path[0]  # or loop over ugv_path if needed

                    traversed = np.where(env.state['UAV_STATE']['traversal_status'] != 0)[0]
                    
                    if env.state['UAV_STATE']['traversal_status'][ugv_index] is None:
                        action[ugv_index + 6] = 1
                    else:
                        if ugv_index + 7 < len(action):
                            action[ugv_index + 1 + 6] = 1
                        else:
                            action[ugv_index - 1 + 6] = 1

            state, reward, done, _, info = env.step(action)

            traversal_status = env.state['UGV_STATE']['traversal_status']
            last_move_idx = np.where(env.state['UGV_STATE']['last_move'] == 1)[0]
            
            if len(last_move_idx) > 0:
                mines_encountered = 1 if traversal_status[last_move_idx[0]] == 2 else 0
                total_mines += mines_encountered
        
        initial_metrics_df = env.get_metrics_dataframe()
        print(initial_metrics_df['mines_encountered'].values[0])

        env.save(os.path.join(save_dir, f"out_human_multi_{i}.csv"))
        initial_metrics_df["eval"] = True
        eval_metrics_list.append(initial_metrics_df)

    metrics_df = pd.concat(eval_metrics_list)
    initial_eval_csv = os.path.join(save_dir, f"eval_metrics.csv")
    metrics_df.to_csv(initial_eval_csv, index=False)
       
    print(total_mines)

def eval(timestamp = None):
    """
    Trains a PPO agent with periodic evaluation.
    Before training begins, an initial evaluation episode is run (with eval=True)
    and its metrics are logged to CSV. Then training continues with periodic evaluation
    episodes. A folder with the timestamp is created to store results, and whenever
    the last evaluation reports 0 mines encountered, the model is saved into a 'saves'
    subfolder.
    """
    import datetime
    import os
    import pandas as pd
    from stable_baselines3 import PPO  # type: ignore
    save_dir = 'results/results/runs'
    # Initialize the environment (training mode by default)
    env = Environment(save_dir=save_dir)
    
    # Initialize the PPO model
    model = PPO.load("results/2025-04-11_12-36/saves/15376.zip", env=env)
    
    eval_metrics_list = []
    
    for i in range(10):
        obs, _ = env.reset(eval=True)
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
        
        initial_metrics_df = env.get_metrics_dataframe()
        print(initial_metrics_df['mines_encountered'].values[0])
    
        env.save(os.path.join(save_dir, f"out_{i}.csv"))
        initial_metrics_df["eval"] = True
        eval_metrics_list.append(initial_metrics_df)

    metrics_df = pd.concat(eval_metrics_list)
    initial_eval_csv = os.path.join(save_dir, f"eval_metrics.csv")
    metrics_df.to_csv(initial_eval_csv, index=False)
if __name__ == '__main__':
    #main()
    #main()
    train()
    #eval()
  