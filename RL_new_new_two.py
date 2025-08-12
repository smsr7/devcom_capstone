import networkx as nx
import numpy as np
import pandas as pd
import gymnasium as gym
from base.missionTwo import Mission
from wrappers.missionWrapperTwo import MissionWrapper
from wrappers.timeKeeper import TimeKeeper
from prediction.predictor import LinearRegressor


class Environment(MissionWrapper, gym.Env):
    def __init__(self, save_dir=None):
        """
        Gym environment wrapping your MissionWrapper with:
          - primary UAV (move + human/AI scan)
          - scanner UAV (move + high-AI scan)
          - automatic UGV along shortest path
        """
        # depths & sizes
        uav_depth = ugv_depth = 1
        MAX_NODES = 6
        self.MAX_NODES = MAX_NODES

        # load regressors for metadata prediction
        self.aiRegressor    = LinearRegressor(target_variable='ai_pred')
        self.humanRegressor = LinearRegressor(target_variable='huma_pred')
        self.aiRegressor.load_model('prediction/ai_regressor_model.pkl')
        self.humanRegressor.load_model('prediction/huma_regressor_model.pkl')

        # initialize base wrapper
        super().__init__(
            uav_depth=uav_depth,
            ugv_depth=ugv_depth,
            max_nodes=MAX_NODES,
            timekeeper=False
        )

        M = self.MAX_NODES

        obs_dim = 78#uav_feat_len + scanner_feat_len + raw_action_len + path_conf_len
        self.action_space = gym.spaces.MultiDiscrete([M, 2, M])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # internal state
        self.state           = None
        self.previous_action = np.zeros(self.MAX_NODES * 3, dtype=int)
        self.save_dir        = save_dir
        self.run_counter     = 0
        self.metrics_list    = []

    def make_prediction(self, metadata):
        """
        Same as before: uses regressors to predict ai/human estimates
        based on environmental metadata.
        """
        df = pd.json_normalize(metadata)
        num_vars = ['temperature','wind_speed','visibility','precipitation']
        df[num_vars] = df[num_vars].fillna(0)
        if 'mines' not in df: df['mines'] = 1

        df_no_mine = df.copy(); df_no_mine['mines'] = 1
        df_mine    = df.copy(); df_mine['mines']    = 2
        feats      = ['temperature','wind_speed','visibility','precipitation','mines']

        pred_ai_no    = self.aiRegressor.predict(df_no_mine[feats])
        pred_human_no = self.humanRegressor.predict(df_no_mine[feats])
        pred_ai_yes   = self.aiRegressor.predict(df_mine[feats])
        pred_human_yes= self.humanRegressor.predict(df_mine[feats])

        results = []
        for i in range(len(df)):
            results.append({
                'ai_pred_no_mine':    pred_ai_no[i],
                'human_pred_no_mine': pred_human_no[i],
                'ai_pred_mine':       pred_ai_yes[i],
                'human_pred_mine':    pred_human_yes[i]
            })
        out = pd.concat([df, pd.DataFrame(results)], axis=1)
        mask = out['terrain']=='unknown'
        out.loc[mask, ['ai_pred_no_mine','human_pred_no_mine',
                       'ai_pred_mine','human_pred_mine']] = 0
        return out

    def _to_one_hot(self, uav_dir, scan_flag, scanner_dir):
        M = self.MAX_NODES
        # primary UAV: 2*M long
        primary = np.zeros(2*M, dtype=int)
        if scan_flag:
            primary[uav_dir] = 1            # AI slots [0..M-1]
        else:
            primary[M + uav_dir] = 1        # human slots [M..2M-1]

        # scanner UAV: M long
        scanner = np.zeros(M, dtype=int)
        scanner[scanner_dir] = 1

        return primary, scanner


    def _process_scanner_state(self, state):
        ss       = state['SCANNER_UAV_STATE']
        move     = np.array(ss['move_array'], dtype=float)
        path_dir = float(np.argmax(ss['path_array']))
        last_dir = float(np.argmax(ss['last_move']))
        return np.concatenate([move, [path_dir, last_dir]]).astype(np.float32)

    def _process_uav_state(self, state):
        """
        Primary UAV features + scanner-UAV features + UGV next-move from UGV_STATE.
        """
        us = state['UAV_STATE']
        gs = state['UGV_STATE']

        def nz(arr):
            return np.array([0 if v is None else v for v in arr], dtype=float)

        # 1) UAV-local arrays
        trav_arr = nz(us['traversal'])         # shape: (M,)
        move_arr = np.array(us['move_array'])  # shape: (M,)
        path_arr = np.array(us['path_array'])  # shape: (M,)
        last_arr = np.array(us['last_move'])   # shape: (M,)

        # 2) UGV next-move one-hot
        ugv_dir_arr = np.array(gs['path_array'], dtype=int)  # shape: (M,)

        # 3) Discretize the one-hot directions
        path_dir = float(np.argmax(path_arr))
        last_dir = float(np.argmax(last_arr))
        ugv_dir  = float(np.argmax(ugv_dir_arr))

        # 4) Build primary-UAV feature vector
        #    [trav_arr, path_dir, last_dir, ugv_dir, move_arr,
        #     num_connections, uav_dist_to_goal, ugv_dist_to_goal]
        uav_vec = np.concatenate([
            trav_arr,                         # M
            [path_dir, last_dir, ugv_dir],   # 3
            move_arr,                         # M
            [
                us['num_connections'],        # 1
                us['uav_distance_to_goal'],   # 1
                us['ugv_distance_to_goal']    # 1
            ]
        ]).astype(np.float32)                 # length = 2M + 6

        # 5) Append scanner-UAV features
        scanner_vec = self._process_scanner_state(state)  # length = M + 2

        return np.concatenate([uav_vec, scanner_vec])   # length = 2M+6 + M+2

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

    def reset(self, delay_start=20, max_iter=400, max_scans=300, regen=False, eval_mode=False, test=False, **kwargs):
        """
        Resets the environment and returns the first observation of the correct shape:
        [uav_feats, scanner_feats, raw_action(3), path_confidence(1), last_state(history...)]
        """
        # 1) Call parent reset to re-generate the network/mission
        raw_state, _ = super().reset(eval=eval_mode)
        self.state = raw_state

        # 2) Reset all your per-episode counters & flags
        self.iter = 0
        self.move_ugv = False
        self.max_scans = max_scans
        self.uav_goal = False
        self.previous_ugv_distance = self.state['UAV_STATE']['ugv_distance_to_goal']
        self.cumulative_ugv_weights = None
        self.path_confidence = 50.0

        # 3) Reset raw-action/history memory
        self.prev_uav_dir = 0
        self.prev_scan_flag = 0
        self.prev_scanner_dir = 0

        # 4) Build the “core” observation (exactly as in step())
        uav_feats       = self._process_uav_state(self.state)
        scanner_feats   = self._process_scanner_state(self.state)
        raw_action_vec  = np.array([0, 0, 0], dtype=np.float32)
        conf_vec        = np.array([self.path_confidence, self.iter], dtype=np.float32)

        core_obs = np.concatenate([
            uav_feats,
            scanner_feats,
            raw_action_vec,
            conf_vec
        ]).astype(np.float32)

        # 5) Initialize last_state to match shape (zeros)
        #    In step() you do: next_obs = np.concatenate([core_obs, self.last_state])
        #    so here we make last_state be zeros of the same length as core_obs
        self.last_state = np.zeros_like(core_obs, dtype=np.float32)

        # 6) Return the combined observation
        obs = np.concatenate([core_obs, self.last_state])

        self.current_episode_metrics = {
            'run':             self.run_counter,
            'actions':         [],
            'iterations':      0,
            'ugv_start':       None,
            'mines_encountered': 0,
            'total_time':      0,
            'reward':          0,
            'path_confidence': 0
        }

        self.eval_step_data = []

        return obs, None

    def step(self, action):
        """
        action: [uav_direction (0..M-1), scan_flag (0=human,1=AI), scanner_direction (0..M-1)]
        Returns: obs, reward, done, info
        """
        M = self.MAX_NODES

        # 1) Unpack the raw action triple
        uav_dir, scan_flag, scanner_dir = map(int, action)

        # 2) Convert to one-hot vectors for each UAV
        primary_onehot, scanner_onehot = self._to_one_hot(uav_dir, scan_flag, scanner_dir)

        if self.eval_mode:
            su = self.uav_node
            sg = self.ugv_node
            sc = self.scanner_uav_node

        # 3) Step the primary UAV (move + human/AI scan)
        self.step_uav(primary_onehot)

        # 3a) Update weight from primary UAV estimate
        if self.last_uav_edge is not None:
            uav_state = self.get_uav_state()
            idx       = int(np.argmax(uav_state['last_move']))
            human_est = uav_state['human_estimates'][idx]
            ai_est    = uav_state['ai_estimates'][idx]
            if human_est is not None:
                self.update_edge_weight(self.last_uav_edge, 100 * human_est)
            elif ai_est is not None:
                self.update_edge_weight(self.last_uav_edge, 100 * ai_est)

        # 4) Step the scanner UAV (move + high-AI scan)
        self.step_scanner_uav(scanner_onehot)

        # 4a) Update weight from scanner UAV’s high-AI estimate
        if getattr(self, 'last_scanner_uav_edge', None) is not None:
            sc_state    = self.get_scanner_uav_state()
            idx         = int(np.argmax(sc_state['last_move']))
            high_ai_est = sc_state['ai_estimates'][idx]
            if high_ai_est is not None:
                self.update_edge_weight(self.last_scanner_uav_edge, 100 * high_ai_est)
            
        # 5a) Append the last edge weight into cumulative_ugv_weights
            if self.last_ugv_edge is not None:
                w = self.network[self.last_ugv_edge[0]][self.last_ugv_edge[1]]['weight']
                if self.cumulative_ugv_weights is None:
                    self.cumulative_ugv_weights = np.array([w], dtype=float)
                else:
                    self.cumulative_ugv_weights = np.append(self.cumulative_ugv_weights, w)

            # 5b) Compute remaining path weights
            try:
                path_nodes = nx.shortest_path(
                    self.network, source=self.ugv_node, target=self.GOAL, weight='weight'
                )
            except nx.NetworkXNoPath:
                path_nodes = []
 
            if len(path_nodes) > 1:
                rem_weights = np.array([
                    self.network[u][v]['weight']
                    for u, v in zip(path_nodes[:-1], path_nodes[1:])
                ], dtype=float)
            else:
                rem_weights = np.array([], dtype=float)

            # 5c) Build a proper 1D array for historic weights
            if self.cumulative_ugv_weights is None:
                hist_weights = np.array([], dtype=float)
            else:
                hist_weights = np.array(self.cumulative_ugv_weights, dtype=float)

            # 5d) Concatenate and average
            all_weights = np.concatenate([hist_weights, rem_weights])
            if all_weights.size > 0:
                self.path_confidence = float(all_weights.mean())
            else:
                self.path_confidence = 50.0


        if self.move_ugv:
            ugv_state = self.get_ugv_state()
            ugv_path  = np.array(ugv_state['path_array'], dtype=int)
            if ugv_path.any():
                ugv_move = np.zeros(M, dtype=int)
                ugv_move[np.argmax(ugv_path)] = 1
                self.step_ugv(ugv_move)

        # 5e) Decide whether to begin auto-moving
        start_cond = (
            self.path_confidence < 35
            or self.iter > self.max_scans
            or (getattr(self, 'test', False) and self.iter > self.delay_start)
        )
        
        if start_cond and not self.move_ugv:
            self.current_episode_metrics['ugv_start'] = self.iter
            self.move_ugv = True

        # 6) Record the raw action for the next observation
        self.prev_uav_dir     = uav_dir
        self.prev_scan_flag   = scan_flag
        self.prev_scanner_dir = scanner_dir

        # 7) Build the next observation vector
        self.state       = self.get_state()
        uav_feats       = self._process_uav_state(self.state)
        scanner_feats   = self._process_scanner_state(self.state)
        raw_action_vec  = np.array([uav_dir, scan_flag, scanner_dir], dtype=np.float32)
        confidence_vec  = np.array([self.path_confidence, self.iter], dtype=np.float32)

        obs = np.concatenate([
            uav_feats,
            scanner_feats,
            raw_action_vec,
            confidence_vec
        ])

        # 8) Compute reward and done flag
        done   = (self.ugv_node == self.GOAL)
        reward = self.calculate_reward(
            self.state,
            (uav_dir, scan_flag, scanner_dir),
            done
        )

        # 9) Advance iteration counter
        self.iter += 1
        self.current_episode_metrics['iterations'] = self.iter
        self.current_episode_metrics['actions'] = (uav_dir, scan_flag, scanner_dir)
        self.current_episode_metrics['reward'] += reward

        # check if UGV just encountered a mine
        ugv_state      = self.state['UGV_STATE']
        last_move_one  = np.array(ugv_state['last_move'], dtype=int)
        if last_move_one.any():
            idx = int(np.argmax(last_move_one))
            if ugv_state['traversal'][idx] == 2:
                self.current_episode_metrics['mines_encountered'] += 1

        # capture ugv_start if triggered this step
        if self.move_ugv and self.current_episode_metrics['ugv_start'] is None:
            self.current_episode_metrics['ugv_start'] = self.iter

        # record total_time & path_confidence
        self.current_episode_metrics['total_time'] = self.get_time()
        self.current_episode_metrics['path_confidence'] = self.path_confidence

        if self.eval_mode:
            self.eval_step_data.append({
                "action": action,
                "start_uav": su,
                "start_ugv": sg,
                "start_scanner": sc,
                "end_uav":   self.uav_node,
                "end_ugv":   self.ugv_node,
                "end_scanner": self.scanner_uav_node
                })

        # at episode end, flush metrics
        if done:
            self.metrics_list.append(self.current_episode_metrics)
            self.run_counter += 1

        state = np.concatenate([obs.copy(), self.last_state])
        
        self.last_state = obs.copy()

        return state, reward, done, {}, {}
        #return obs, reward, terminated, truncated, info

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

    def get_metrics_dataframe(self):
        """Return a DataFrame of all completed runs."""
        df = pd.DataFrame(self.metrics_list)
        df.set_index('run', inplace=True)
        # reset for next invocation
        self.metrics_list = []
        return df

    def get_state(self):
        """
        Return internal wrapper state dict.
        """
        return {
            'UAV_STATE':         self.get_uav_state(),
            'SCANNER_UAV_STATE': self.get_scanner_uav_state(),
            'UGV_STATE':         self.get_ugv_state()
        }

    # ─── inside Environment ──────────────────────────────────────────────────

    def calculate_reward(self, state, action, done):
        """
        Compute the reward for the current state and action, including:
        - Duplicate-scan penalty
        - UAV distance progress
        - Scan confidence
        - Goal bonus
        - Path-confidence reward
        """
        # Unpack
        uav_dir, scan_flag, scanner_dir = action

        # 1) Duplicate‐scan penalty
        DUP_PENALTY = -10.0
        reward = 0.0

        # — primary UAV —
        last_u = np.array(state['UAV_STATE']['last_move'], dtype=int)
        if last_u.any():
            idx = int(np.argmax(last_u))
            edge = tuple(state['UAV_STATE']['edges'][idx])
            eid  = tuple(sorted(edge))
            mask = self.scan_status.get(eid, 0)
            if scan_flag == 1 and (mask & 1):  # AI bit
                reward += DUP_PENALTY
            if scan_flag == 0 and (mask & 2):  # human bit
                reward += DUP_PENALTY

        # — scanner UAV —
        last_s = np.array(state['SCANNER_UAV_STATE']['last_move'], dtype=int)
        if last_s.any():
            idx2  = int(np.argmax(last_s))
            edge2 = tuple(state['SCANNER_UAV_STATE']['edges'][idx2])
            eid2  = tuple(sorted(edge2))
            mask2 = self.scan_status.get(eid2, 0)
            if (mask2 & 4):  # high-AI bit
                reward += DUP_PENALTY

        uav_dist = state['UAV_STATE']['uav_distance_to_goal']
        scanner_dist = state['SCANNER_UAV_STATE']['scanner_distance_to_goal']
            # distances
        uav_dist     = state['UAV_STATE']['uav_distance_to_goal']
        scanner_dist = state['SCANNER_UAV_STATE']['scanner_distance_to_goal']

        # 2) UAV distance‐progress + best‐distance bonus
        if not getattr(self, 'uav_goal', False):
            dist_mult       = 1.0
            best_mult       = 3
            prev_dist       = getattr(self, 'old_uav_distance', uav_dist)
            # progress
            if uav_dist > 0:
                prog = 100 * (prev_dist - uav_dist) / uav_dist
            else:
                prog = 0.0
            reward += dist_mult * prog
            # best‐distance
            if hasattr(self, 'best_uav_distance') and self.best_uav_distance > uav_dist and uav_dist > 0:
                bonus = 100 * (self.best_uav_distance - uav_dist) / uav_dist
                reward += best_mult * bonus
                self.best_uav_distance = uav_dist
            else:
                # first time or no improvement yet
                self.best_uav_distance = uav_dist
            # update old
            self.old_uav_distance = uav_dist

        # 3) Scanner distance‐progress + best‐distance bonus
        if not getattr(self, 'scanner_goal', False):
            dist_mult       = 1.0
            best_mult_sc    = 3
            prev_scanner    = getattr(self, 'old_scanner_distance', scanner_dist)
            if scanner_dist > 0:
                prog_s = 100 * (prev_scanner - scanner_dist) / scanner_dist
            else:
                prog_s = 0.0
            reward += dist_mult * prog_s
            if hasattr(self, 'best_scanner_distance') and self.best_scanner_distance > scanner_dist and scanner_dist > 0:
                bonus_s = 100 * (self.best_scanner_distance - scanner_dist) / scanner_dist
                reward += best_mult_sc * bonus_s
                self.best_scanner_distance = scanner_dist
            else:
                self.best_scanner_distance = scanner_dist
            self.old_scanner_distance = scanner_dist


        # 3) Scan‐confidence reward
        radius_mult = 0
        last_idx = int(np.argmax(state['UAV_STATE']['last_move']))
        ai_pred   = state['UAV_STATE']['ai_estimates'][last_idx]
        human_pred= state['UAV_STATE']['human_estimates'][last_idx]
        preds = [p for p in (ai_pred, human_pred) if p is not None]
        scan_conf = np.mean(np.abs(np.array(preds) - 0.5)) if preds else 0.0

        reward += radius_mult * scan_conf

        # 4) UAV‐goal bonus
        if uav_dist == 0 and not getattr(self, 'uav_goal', True):
            reward += 1000.0
            self.uav_goal = True
            print("UAV_GOAL")

        if scanner_dist == 0 and not getattr(self, 'scanner_goal', True):
            reward += 1000.0
            self.scanner_goal = True
            print("SCANNER_GOAL")

        # 5) Path‐confidence reward
        path_confidence_weight = 1
        pc = getattr(self, 'path_confidence', 50.0)
        path_confidence_reward = (35.0 - pc)
        # big bonus if it’s very low
        if pc < 30.0:
            reward += 100.0

        reward += path_confidence_weight * path_confidence_reward

        #reward -= self.iter - 200 if self.iter > 200 else 0

        # — normalize & return —
        return reward / 100.0


# ─── outside the class ───────────────────────────────────────────────────────

import datetime
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def train(timestamp=None, model_path=None):
    n_eval = 1
    total_timesteps = 50_000_000
    eval_interval   = 10_240

    # folders
    now = datetime.datetime.now()
    ts = timestamp or now.strftime('%Y-%m-%d_%H-%M')
    results_folder = os.path.join("results", ts)
    os.makedirs(results_folder, exist_ok=True)
    saves_folder = os.path.join(results_folder, "saves")
    os.makedirs(saves_folder, exist_ok=True)

    # raw env (for eval)
    raw_env = Environment(save_dir=ts)

    # training env
    def make_env():
        return Environment(save_dir=ts)
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, gamma=0.99)

    # model
    if model_path:
        model = PPO.load(model_path, env=train_env)
    else:
        model = PPO("MlpPolicy", train_env, verbose=1)

    all_metrics = pd.DataFrame()

    def normalize(o):
        mean = train_env.obs_rms.mean
        var  = train_env.obs_rms.var
        eps  = train_env.epsilon
        clip = train_env.clip_obs
        return np.clip((o - mean)/np.sqrt(var+eps), -clip, clip)

    # initial eval
    raw_obs, _ = raw_env.reset(eval_mode=True)
    done = False
    obs = normalize(raw_obs)
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        raw_obs, _, done, _, _ = raw_env.step(act)
        obs = normalize(raw_obs)

    df0 = raw_env.get_metrics_dataframe()
    df0["eval"] = False
    df0.iloc[-1, df0.columns.get_loc("eval")] = True
    df0["timesteps"] = 0
    all_metrics = pd.concat([all_metrics, df0], ignore_index=True)

    all_metrics.to_csv(os.path.join(results_folder,
                                        f"eval_metrics_{ts}.csv"),
                           index=False)
    raw_env.save(os.path.join(saves_folder, f"out_init.csv"))

    
    timesteps = 0
    while timesteps < total_timesteps:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        timesteps += eval_interval

        # evaluation
        for _ in range(n_eval):
            raw_obs, _ = raw_env.reset(eval_mode=True)
            done = False
            obs = normalize(raw_obs)
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                raw_obs, _, done, _, _ = raw_env.step(act)
                obs = normalize(raw_obs)

        df = raw_env.get_metrics_dataframe()
        df["eval"] = False
        df.iloc[-n_eval:, df.columns.get_loc("eval")] = True
        df["timesteps"] = timesteps
        all_metrics = pd.concat([all_metrics, df], ignore_index=True)
        all_metrics.to_csv(os.path.join(results_folder,
                                        f"eval_metrics_{ts}.csv"),
                           index=False)

        # save model and metrics
        if (df["mines_encountered"] <= 4).any():
            model.save(os.path.join(saves_folder, f"{timesteps}.zip"))
        raw_env.save(os.path.join(saves_folder, f"out_{timesteps}.csv"))

    # final test
    raw_obs, _ = raw_env.reset(eval_mode=False)
    done = False
    obs = normalize(raw_obs)
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        raw_obs, _, done, _, _ = raw_env.step(act)
        obs = normalize(raw_obs)
    df_final = raw_env.get_metrics_dataframe()
    df_final["eval"] = False
    df_final["timesteps"] = timesteps
    df_final.to_csv(os.path.join(results_folder,
                                 f"final_metrics_{ts}.csv"),
                    index=False)

def main():
    """
    Baseline run: primary UAV and scanner UAV follow their own shortest paths.
    UGV auto-moves (env.move_ugv=True).
    """
    import os
    save_dir = 'results/baselines'
    os.makedirs(save_dir, exist_ok=True)

    env = Environment()
    obs, _ = env.reset(eval_mode=True, test=True)

    # Turn on auto-UGV navigation
    env.move_ugv = False

    done = False
    total_mines = 0
    M = env.MAX_NODES
    reward = 0
    time = 0
    while not done:
        time += 1
        # 1) Primary UAV next‐move direction
        uav_path_array = np.array(env.state['UAV_STATE']['path_array'], dtype=int)
        if uav_path_array.any():
            uav_direction = int(np.argmax(uav_path_array))
            uav_direction = uav_direction if uav_direction < env.MAX_NODES * 2 else uav_direction - 1
        else:
            uav_direction = np.random.randint(0, M)

        # 2) Decide scan_flag (0=human, 1=AI) — here, always use AI for baseline
        scan_flag = 1

        # 3) Scanner UAV next‐move direction
        scanner_path_array = np.array(env.state['SCANNER_UAV_STATE']['path_array'], dtype=int)
        if scanner_path_array.any():
            if time > 2:
                scanner_direction = int(np.argmax(scanner_path_array))
            else:
                scanner_direction = int(np.argmax(scanner_path_array)) + 1
        else:
            scanner_direction = np.random.randint(0, M)

        print(env.iter, reward, env.path_confidence, env.move_ugv, env.state['UGV_STATE']['ugv_distance_to_goal'], env.state['UGV_STATE']['path_array'])
        # 4) Step the environment with the raw triple action
        raw_action = [uav_direction, scan_flag, scanner_direction]
        obs, reward, done, _, info = env.step(raw_action)

        # 5) Tally mines encountered by UGV
        ugv_state        = env.state['UGV_STATE']
        traversal_flags  = ugv_state['traversal']
        last_move_onehot = np.array(ugv_state['last_move'], dtype=int)
        if last_move_onehot.any():
            last_idx = int(np.argmax(last_move_onehot))
            if traversal_flags[last_idx] == 2:
                total_mines += 1

    # 6) Save results
    df = env.get_metrics_dataframe()
    df["eval"] = True
    df.to_csv(os.path.join(save_dir, "baseline_ai.csv"), index=False, mode='a')
    print(f"Done: encountered {total_mines} mines.")
    env.save(os.path.join(save_dir, "baseline_ai_out.csv"))




if __name__ == '__main__':
    train()
