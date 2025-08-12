from wrappers.missionWrapper import MissionWrapper
from prediction.predictor import LinearRegressor
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import networkx as nx  # type: ignore


class Environment(MissionWrapper):

    def __init__(self):
        uav_depth = 2
        ugv_depth = 2

        MAX_NODES = 3

        self.aiRegressor = LinearRegressor(target_variable='ai_pred')
        self.humanRegressor = LinearRegressor(target_variable='huma_pred')

        super().__init__(uav_depth=uav_depth, ugv_depth=ugv_depth, max_nodes=MAX_NODES, timekeeper=False)

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

    def _process_uav_state(self, state):
        metadata_state = self.make_prediction(state['UAV_STATE']['metadata'])

        def replace_none_with_zero(array):
            return np.array([0 if x is None else x for x in array])

        traversal_status = replace_none_with_zero(state['UAV_STATE']['traversal_status'])
        ai_estimates = replace_none_with_zero(state['UAV_STATE']['ai_estimates'])
        human_estimates = replace_none_with_zero(state['UAV_STATE']['human_estimates'])

        path_array = np.array(state['UAV_STATE']['path_array'])
        move_array = np.array(state['UAV_STATE']['move_array'])
        last_move = np.array(state['UAV_STATE']['last_move'])
        ugv_path_array = np.array(state['UAV_STATE']['ugv_path_array'])

        metadata_state_no_time = metadata_state.drop(columns=['time', 'terrain'])

        metadata_values = metadata_state_no_time.values.flatten()

        state = np.concatenate([
            traversal_status,
            ai_estimates,
            human_estimates,
            path_array,
            move_array,
            last_move,
            [state['UAV_STATE']['num_connections'],
                state['UAV_STATE']['uav_distance_to_goal'],
                state['UAV_STATE']['ugv_distance_to_goal']],
            ugv_path_array,
            metadata_values
        ])

        return state

    def update_edge_weight(self, edge, weight):
        u, v = edge
        if self.network.has_edge(u, v):
            self.network[u][v]['weight'] = weight
            self.network[v][u]['weight'] = weight
        else:
            print(f"Edge ({u}, {v}) not found in the network.")

    def reset(self):
        state = super().reset()

        self.aiRegressor.load_model('prediction/ai_regressor_model.pkl')
        self.humanRegressor.load_model('prediction/huma_regressor_model.pkl')

        self.state = state

        state = self._process_uav_state(self.state)
        return state

    def step(self, action):
        done = False

        state, done = super().step(action)
        self.update_edge_weight(self.last_uav_edge, 5)  #  NEED TO UPDATE HOW WE ARE SETTING SCANNED CELLS

        state = self.get_state()

        self.state = state
        state = self._process_uav_state(self.state)

        return state, done


def main():
    #  code to go straight to goal
    env = Environment()
    env.reset()
    done = False

    #  move the uav first and
    action = np.zeros(env.MAX_NODES * 4)
    action[np.where(env.state['UAV_STATE']['path_array'] == 1)[0] + 1*env.MAX_NODES] = 1
    state, done = env.step(action)

    while not done:
        action = np.zeros(env.MAX_NODES * 4)

        action[np.where(env.state['UGV_STATE']['path_array'] == 1)[0] + 3*env.MAX_NODES] = 1
        action[np.where(env.state['UAV_STATE']['path_array'] == 1)[0] + 1*env.MAX_NODES] = 1
        state, done = env.step(action)

    #  state, done = env.step([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    #  env.pretty_state()


if __name__ == '__main__':
    main()
