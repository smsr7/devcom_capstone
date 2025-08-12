from missionWrapper import MissionWrapper


class Environment(MissionWrapper):

    def __init__(self):
        uav_depth = 2
        ugv_depth = 2

        MAX_NODES = 3

        super().__init__(uav_depth=uav_depth, ugv_depth=ugv_depth, max_nodes=MAX_NODES, timekeeper=False)

    def reset(self):
        self.state = super().reset()

        return self.state

    def step(self, action):
        done = False

        state, done = super().step(action)
        self.state = state

        return self.state, done


if __name__ == '__main__':
    env = Environment()
    env.reset()
    env.pretty_state()
    state, done = env.step([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    env.pretty_state()
