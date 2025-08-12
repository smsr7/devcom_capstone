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
        self.MAX_NODES = MAX_NODES  # make sure to assign it as an instance variable

        self.BASE_WEIGHT = 50

        self.aiRegressor = LinearRegressor(target_variable='ai_pred')
        self.humanRegressor = LinearRegressor(target_variable='huma_pred')

        self.aiRegressor.load_model('prediction/ai_regressor_model.pkl')
        self.humanRegressor.load_model('prediction/huma_regressor_model.pkl')

        super().__init__(uav_depth=uav_depth, ugv_depth=ugv_depth, max_nodes=MAX_NODES, timekeeper=False)

        # ----------------------- NEW/UPDATED -----------------------
        # Instead of a MultiBinary space, we now use a Tuple space so that the
        # agent outputs (direction, scan_flag).  
        # Direction: Discrete(MAX_NODES) – an integer in 0...MAX_NODES-1
        # Scan: Discrete(2) – 0 or 1 (for example, 0 = use UGV scan; 1 = use UAV scan)
        self.action_space = gym.spaces.MultiDiscrete([self.MAX_NODES, 2])

        # -----------------------------------------------------------

        # Ovservation space: This was determined by running RLagent.py
        observation_shape = (64,)
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


    def onehot_to_index(self, onehot_array):
        """
        Converts a one-hot vector (length 6) into its integer index (0–5).
        """
        return float(np.argmax(onehot_array))


    def _process_uav_state(self, state):
        
        metadata_state = self.make_prediction(state['UAV_STATE']['metadata'])

        def replace_none_with_zero(array):
            return np.array([0 if x is None else x for x in array])

        traversal_status = replace_none_with_zero(state['UAV_STATE']['traversal_status'])
        move_array       = np.array(state['UAV_STATE']['move_array'])

        # get the three one-hots
        path_array      = np.array(state['UAV_STATE']['path_array'])
        last_move       = np.array(state['UAV_STATE']['last_move'])
        ugv_path_array  = np.array(state['UAV_STATE']['ugv_path_array'])

        # convert each into a discrete index
        path_dir = self.onehot_to_index(path_array)
        last_dir = self.onehot_to_index(last_move)
        ugv_dir  = self.onehot_to_index(ugv_path_array)

        # pack your continuous metadata as before
        metadata_state_no_time = metadata_state.drop(columns=['time', 'terrain'])
        metadata_values        = metadata_state_no_time.values.flatten()

        # now build the final state vector
        # order it however makes sense—here’s one example:
        #print(len(traversal_status), path_dir, last_dir, ugv_dir)
        state_vector = np.concatenate([
            traversal_status,           # e.g. 6 floats
            [path_dir, last_dir, ugv_dir],  # three discrete floats 0–5
            move_array,                 # e.g. 6 floats
            #metadata_values,            # however many floats your metadata has
            [                           # your other scalars
                state['UAV_STATE']['num_connections'],
                state['UAV_STATE']['uav_distance_to_goal'],
                state['UAV_STATE']['ugv_distance_to_goal']
            ]
        ]).astype(np.float32)

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
            output_json_path = "config/network_test_two.json"              # Path to save the generated network
            map_generator = MapGenerator(parameters_file, output_json_path)
            G, start_node, end_node = map_generator.generate_network()

        self.state = state
        self.previous_ugv_distance = self.state['UAV_STATE']['ugv_distance_to_goal']

        state = self._process_uav_state(self.state)
        self.last_state = np.concatenate([state.copy(), np.zeros(14)])

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
        self.avg_weight = 50

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
        self.best_distance = self.state['UAV_STATE']['uav_distance_to_goal']
        self.old_distance = self.state['UAV_STATE']['uav_distance_to_goal']

        return obs, None

    def step(self, action):
        done = False
        if (isinstance(action, (tuple, list)) and len(action) == 2) or (isinstance(action, np.ndarray) and action.shape[0] == 2):
            # Extract and round the outputs (in case the outputs are probabilities or continuous values)
            direction_index = int(round(action[0]))
            scan_flag = bool(round(action[1]))
            # Ensure the index is in bounds:
            if direction_index < 0 or direction_index >= self.MAX_NODES:
                direction_index = np.random.randint(0, self.MAX_NODES)
            # Create the one-hot vector for the UAV part:
            uav_action = np.zeros(self.MAX_NODES, dtype=int)
            uav_action[direction_index] = 1
            # For the scan action:
            if scan_flag:
                # For example: if scan_flag is True, we can choose to have NO UGV scan activated
                ugv_action = np.zeros(self.MAX_NODES, dtype=int)
            else:
                # Otherwise, if scan_flag is False, use the current state's UGV path indicator.
                ugv_action = np.zeros(self.MAX_NODES, dtype=int)
                ugv_path_array = np.array(self.state['UGV_STATE']['path_array'])
                indices = np.where(ugv_path_array == 1)[0]
                if len(indices) > 0:
                    ugv_action[indices[0]] = 1
                else:
                    ugv_action[0] = 1
            # Construct final one-hot encoded action vector:
            action = np.concatenate([uav_action, ugv_action])
        # ------------------------------------------------------------------------------

        out_action = action.copy()

        # Get the UAV state's move array and duplicate it (as before)
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
        obs = np.concatenate([obs, np.asarray([self.iter, int(self.avg_weight)])])

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
        #reward += err_reward
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
        (Implementation remains unchanged.)
        """
        # --- Existing Reward Components ---
        dist_mult = 1
        path_confidence_weight = 0.5
        radius_mult = 1
        bomb_mult = 0
        action_mult = 5
        uav_mult = 0
        uav_best_mult = 0
        confidence_mult = 0

        dupe_reward_val = 1

        reward = 0

        ugv_distance = state['UAV_STATE']['ugv_distance_to_goal']
        previous_ugv_distance = self.previous_ugv_distance

        distance_reward = (abs(previous_ugv_distance - ugv_distance)
                        if self.move_ugv and not done
                        else abs((previous_ugv_distance - ugv_distance) - self.BASE_WEIGHT))
        
        action_reward = -1 * (action.sum() - 1) if action.sum() != 1 else 100
        ai_count = np.count_nonzero(state['UGV_STATE']['ai_estimates'] is not None)
        human_count = np.count_nonzero(state['UGV_STATE']['human_estimates'] is not None)
        radius_reward = (ai_count + human_count)
        
        reward += radius_reward * radius_mult
        reward += distance_reward * dist_mult
        
        traversal_status = state['UGV_STATE']['traversal_status']
        last_move_idx = np.where(state['UGV_STATE']['last_move'] == 1)[0]
        bomb_reward = 0
        self.previous_ugv_distance = ugv_distance
        confidence_reward = self.last_scan_confidence * 100
        '''reward = (dist_mult * distance_reward +
                radius_mult * radius_reward +
                bomb_mult * bomb_reward +
                action_mult * action_reward + 
                confidence_mult * confidence_reward)
        reward -= self.iter * 10 if self.iter > 200 else 0
        reward += dupe_reward_val if dupe_reward else 0'''
        uav_distance = state['UAV_STATE']['uav_distance_to_goal']
        GOAL_BONUS = 100

        if uav_distance == 0:
            print("GOAL_REACHED")
            if not self.uav_goal:
                reward += GOAL_BONUS
            else:
                reward -= 0
            self.uav_goal = True

        if not self.uav_goal:
            uav_reward = ((self.old_distance - uav_distance) ) / uav_distance
            reward += uav_reward * uav_mult

            if self.best_distance > uav_distance:
                uav_best_reward = ((self.best_distance - uav_distance)) / uav_distance
                reward += uav_best_reward * uav_best_mult
                
                self.best_distance = uav_distance

                #print(uav_distance, self.old_distance, self.best_distance, uav_best_reward, uav_reward)

            self.old_distance = uav_distance


        self.old_distance = uav_distance
        path_confidence_reward = (35 - self.avg_weight)
        if self.avg_weight < 25:
            reward += 1000
        reward += path_confidence_reward * path_confidence_weight
        if done:
            self.current_episode_metrics['path_confidence'] = self.avg_weight
        
        return reward / 100
    
    def calculate_reward_old(self, state, action, done, dupe_reward=False):
        """
        Compute the reward for the current state and action.
        (Implementation remains unchanged.)
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

        distance_reward = (abs(previous_ugv_distance - ugv_distance)
                        if self.move_ugv and not done
                        else abs((previous_ugv_distance - ugv_distance) - self.BASE_WEIGHT))
        action_reward = -1 * (action.sum() - 1) if action.sum() != 1 else 100
        ai_count = np.count_nonzero(state['UGV_STATE']['ai_estimates'] is not None)
        human_count = np.count_nonzero(state['UGV_STATE']['human_estimates'] is not None)
        radius_reward = (ai_count + human_count)
        traversal_status = state['UGV_STATE']['traversal_status']
        last_move_idx = np.where(state['UGV_STATE']['last_move'] == 1)[0]
        bomb_reward = 0
        self.previous_ugv_distance = ugv_distance
        confidence_reward = self.last_scan_confidence * 100
        reward = (dist_mult * distance_reward +
                radius_mult * radius_reward +
                bomb_mult * bomb_reward +
                action_mult * action_reward + 
                confidence_mult * confidence_reward)
        reward -= self.iter * 10 if self.iter > 200 else 0
        reward += dupe_reward_val if dupe_reward else 0
        uav_distance = state['UAV_STATE']['uav_distance_to_goal']
        GOAL_BONUS = 10000

        if uav_distance == 0:
            print("GOAL_REACHED")
            if not self.uav_goal:
                reward += GOAL_BONUS
            self.uav_goal = True

        if not self.uav_goal:
            if not hasattr(self, 'best_distance'):
                self.best_distance = state['UAV_STATE']['uav_distance_to_goal']
            if uav_distance < self.best_distance:
                uav_reward = 100 * (self.best_distance - uav_distance)
                reward += uav_reward * uav_mult
                self.best_distance = uav_distance

        self.old_distance = uav_distance
        path_confidence_reward = 100 * (35 - self.avg_weight)
        if self.avg_weight < 25:
            reward += 2500
        reward += path_confidence_reward * path_confidence_weight
        if done:
            self.current_episode_metrics['path_confidence'] = self.avg_weight

        return reward


import datetime
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
def train(timestamp=None, model_path=None):
    """
    Trains a PPO agent with VecNormalize for training.
    If model_path is provided, loads the pretrained model from that path.
    Uses a separate raw env for evaluation (eval=True) and metrics.
    At each checkpoint, runs n_eval eval episodes on the raw env,
    labels only those runs as eval=True, then appends them to the CSV.
    """
    n_eval         = 1       # how many evaluation episodes per checkpoint
    total_timesteps = 50_000_000
    eval_interval   = 10_240

    # 1) Setup folders
    now = datetime.datetime.now()
    if timestamp is None:
        timestamp = now.strftime('%Y-%m-%d_%H-%M')
    results_folder = os.path.join("results", timestamp)
    os.makedirs(results_folder, exist_ok=True)
    saves_folder = os.path.join(results_folder, "saves")
    os.makedirs(saves_folder, exist_ok=True)

    # 2) Create raw env (for evaluation & metrics)
    raw_env = Environment(save_dir=timestamp)

    # 3) Create training env with VecNormalize
    def make_env():
        return Environment(save_dir=timestamp)
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99
    )

    # 4) Instantiate or load model
    if model_path:
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=train_env)
    else:
        model = PPO("MlpPolicy", train_env, verbose=1)

    # 5) Prepare DataFrame to accumulate all runs
    all_metrics_df = pd.DataFrame()

    # Helper: normalize raw observations via VecNormalize stats
    def normalize_obs(o: np.ndarray) -> np.ndarray:
        mean = train_env.obs_rms.mean
        var  = train_env.obs_rms.var
        eps  = train_env.epsilon
        clip = train_env.clip_obs
        norm = (o - mean) / np.sqrt(var + eps)
        return np.clip(norm, -clip, clip)

    # === Initial evaluation before training ===
    print("► Initial evaluation (1 episode)")
    raw_obs, _ = raw_env.reset(eval=True)
    done = False
    obs = normalize_obs(raw_obs)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        raw_obs, _, done, _, _ = raw_env.step(action)
        obs = normalize_obs(raw_obs)
    init_df = raw_env.get_metrics_dataframe()
    print("    Mines encountered:", init_df["mines_encountered"].iloc[-1])
    raw_env.save(os.path.join(saves_folder, "out_initial.csv"))

    # Label only the last row as eval=True
    init_df["eval"] = False
    init_df.iloc[-1, init_df.columns.get_loc("eval")] = True
    init_df["timesteps"] = 0
    all_metrics_df = pd.concat([all_metrics_df, init_df], ignore_index=True)

    # === Main training loop ===
    timesteps = 0
    while timesteps < total_timesteps:
        # Train
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        timesteps += eval_interval

        # Evaluate on raw env
        print(f"\n► Evaluating at {timesteps} timesteps ({n_eval} episode(s))")
        for _ in range(n_eval):
            raw_obs, _ = raw_env.reset(eval=True)
            done = False
            obs = normalize_obs(raw_obs)
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                raw_obs, _, done, _, _ = raw_env.step(action)
                obs = normalize_obs(raw_obs)

        # Collect metrics
        df = raw_env.get_metrics_dataframe()
        # mark only last n_eval rows as eval=True
        df["eval"] = False
        df.iloc[-n_eval:, df.columns.get_loc("eval")] = True
        df["timesteps"] = timesteps
        print("    Recent mines_encountered:", df["mines_encountered"].iloc[-n_eval:].tolist())

        # Optionally save model and scan output
        recent = df["mines_encountered"].iloc[-n_eval:]
        if (recent <= 4).any():
            ckpt_path = os.path.join(saves_folder, f"{timesteps}.zip")
            print("    Saving model to", ckpt_path)
            model.save(ckpt_path)
        raw_env.save(os.path.join(saves_folder, f"out_{timesteps}.csv"))

        # Append and write CSV
        all_metrics_df = pd.concat([all_metrics_df, df], ignore_index=True)
        all_metrics_df.to_csv(
            os.path.join(results_folder, f"eval_metrics_{timestamp}.csv"),
            index=False
        )
        print("    👉 eval_metrics CSV updated")

    # === Final test (no eval flag) ===
    print("\n► Final test (1 episode)")
    raw_obs, _ = raw_env.reset(eval=False)
    done = False
    obs = normalize_obs(raw_obs)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        raw_obs, _, done, _, _ = raw_env.step(action)
        obs = normalize_obs(raw_obs)
    final_df = raw_env.get_metrics_dataframe()
    final_df["eval"] = False
    final_df["timesteps"] = timesteps
    final_df.to_csv(
        os.path.join(results_folder, f"final_metrics_{timestamp}.csv"),
        index=False
    )
    print("Final metrics saved")
#results\2025-04-15_17-47\saves\14875.zip

def main(maxSteps=1000):
    #  code to go straight to goal
    import os
    #  code to go straight to goal
    save_dir = 'results/results/baselines'
    # Initialize the environment (training mode by default)
    eval_metrics_list = []

    for i in range(1):
        env = Environment()
        env.reset(eval=True)
        env.iter = 0
        done = False
        total_mines = 0

        while not done:
            action = np.zeros(env.MAX_NODES * 2)
            
            if not env.uav_goal:
                action[np.where(env.state['UAV_STATE']['path_array'] == 1)[0]] = 1
            else:
                M = env.MAX_NODES
                ugv_idxs = np.where(env.state['UAV_STATE']['ugv_path_array'] == 1)[0]
                if len(ugv_idxs) > 0:
                    idx = ugv_idxs[0]

                    # 2) pull out traversal_status array
                    traversal = env.state['UAV_STATE']['traversal_status']

                    # 3) if that edge already has a traversal status (not None),
                    #    bump to next index (mod M) before moving
                    if traversal[idx] is not None:
                        next_idx = (idx - 1) % M
                        action[next_idx] = 1
                    else:
                        # otherwise, just move at the UGV index
                        action[idx] = 1

            state, reward, done, _, info = env.step(action)
            traversal_status = env.state['UGV_STATE']['traversal_status']
            last_move_idx = np.where(env.state['UGV_STATE']['last_move'] == 1)[0]
            
            if len(last_move_idx) > 0:
                mines_encountered = 1 if traversal_status[last_move_idx[0]] == 2 else 0
                total_mines += mines_encountered
        
        initial_metrics_df = env.get_metrics_dataframe()
        print(initial_metrics_df['mines_encountered'].values[0])

        env.save(os.path.join(save_dir, f"out_ai_multi_{i}.csv"))
        initial_metrics_df["eval"] = True
        eval_metrics_list.append(initial_metrics_df)

    metrics_df = pd.concat(eval_metrics_list)
    initial_eval_csv = os.path.join(save_dir, f"eval_metrics.csv")
    metrics_df.to_csv(initial_eval_csv, index=False, mode='a')
    print(initial_eval_csv)

def ugv():
    
    #  code to go straight to goal
    import os
    #  code to go straight to goal
    save_dir = 'results/results/baselines'
    # Initialize the environment (training mode by default)
    eval_metrics_list = []


    env = Environment()
    env.reset(eval=True)
    env.iter = 0
    done = False
    total_mines = 0

    env.move_ugv = True
    while not done:
        action = np.zeros(env.MAX_NODES * 2)
        state, reward, done, _, info = env.step(action)
        traversal_status = env.state['UGV_STATE']['traversal_status']
        last_move_idx = np.where(env.state['UGV_STATE']['last_move'] == 1)[0]
        
        if len(last_move_idx) > 0:
            mines_encountered = 1 if traversal_status[last_move_idx[0]] == 2 else 0
            total_mines += mines_encountered
    
    initial_metrics_df = env.get_metrics_dataframe()
    print(initial_metrics_df['mines_encountered'].values[0])

# (Other functions train2, main, main2, eval remain unchanged.)
if __name__ == '__main__':
    #main()
    ugv()
    #train(timestamp="singleshot", model_path="results/0_results/race_to_goal/saves/1484800")
    #train2()
    #eval()
