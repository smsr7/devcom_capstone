from missionWrapper import MissionWrapper
import numpy as np


class Environment(MissionWrapper):

    def __init__(self):
        uav_depth = 1
        ugv_depth = 1

        MAX_NODES = 6

        self.metadata = np.array([])
        self.ai_pred = np.array([])
        self.human_pred = np.array([])
        self.mines = np.array([])

        super().__init__(uav_depth=uav_depth, ugv_depth=ugv_depth, max_nodes=MAX_NODES, timekeeper=False)

    def reset(self):
        self.state = super().reset()
        self.order = 0

        return self.state

    def step(self, action):
        done = False

        state, done = super().step(action)
        self.state = state

        return self.state, done

    def get_data(self):
        action = np.zeros(self.MAX_NODES * 4)
        uav_move = np.random.randint(self.MAX_NODES)
        if self.order % 3 == 0:
            action[uav_move + self.MAX_NODES] = 1
            action[uav_move + 3*self.MAX_NODES] = 1

            state, done = self.step(action)

            self.traversal_index = np.where(state['UAV_STATE']['last_move'] == 1)[0][0]
            self.mines = np.append(self.mines, self.get_ugv_state()['traversal_status'][self.traversal_index])

        elif self.order % 3 == 1:
            action[self.traversal_index] = 1
            state, done = self.step(action)
            self.traversal_index = np.where(state['UAV_STATE']['last_move'] == 1)[0][0]

        elif self.order % 3 == 2:
            action[self.traversal_index + 2 * self.MAX_NODES] = 1

            state, done = self.step(action)
            self.traversal_index = np.where(state['UAV_STATE']['last_move'] == 1)[0][0]

            self.ai_pred = np.append(self.ai_pred, state['UAV_STATE']['ai_estimates'][self.traversal_index])
            self.human_pred = np.append(self.human_pred, state['UAV_STATE']['human_estimates'][self.traversal_index])
            self.metadata = np.append(self.metadata, state['UAV_STATE']['metadata'][self.traversal_index])

        self.order += 1


if __name__ == '__main__':
    env = Environment()
    env.reset()
    for i in range(100*env.MAX_NODES):
        env.get_data()
    print(env.traversal_index, env.ai_pred, env.human_pred, env.mines, env.metadata)
