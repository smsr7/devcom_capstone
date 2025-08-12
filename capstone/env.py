from missionWrapper import MissionWrapper


class Environment(MissionWrapper):

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        reward, done = 0, False

        self.get_ugv_state()
        self.get_uav_state()

    def step(self, action):
        done = False

        state, done = super().step(action)
        print(state)


if __name__ == '__main__':
    env = Environment()
    env.reset()
    env.step([0,1,0,0,0,0,0,0,0,
                0,0,0])
