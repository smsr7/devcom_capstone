from wrappers.missionWrapper import MissionWrapper
import networkx as nx  # type: ignore


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
    print(nx.shortest_path(env.network, source=env.ugv_node, target=env.GOAL,
                                  weight='weight'))
