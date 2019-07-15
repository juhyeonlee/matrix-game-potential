import gym
import numpy as np

# from QTRAN code
class MultiAgentSimpleEnv2(gym.Env):  # Matrix game
    def __init__(self):
        self.state = [1]
        self.action_dim = 3
        self.state_dim = 1

        self.payoff2 = np.array([[8., -12., -12.], [-12., 0., 0.], [-12., 0., 0.]])

    def reset(self):
        self.state = [1]

        return self.state

    def step(self, action):
        info = {'n': []}
        reward = []
        done = []
        reward.append(self.payoff2[action[0], action[1]])
        self.state = [3]
        done.append(True)

        return self.state, reward, done, info

    def call_action_dim(self):
        return self.action_dim

    def call_state_dim(self):
        return self.state_dim


# env = MultiAgentSimpleEnv2()
# print(env.step((0, 0)))
# print(env.step((0, 1)))
# print(env.step((0, 2)))
# print(env.step((1, 0)))
# print(env.step((1, 1)))
# print(env.step((1, 2)))
# print(env.step((2, 0)))
# print(env.step((2, 1)))
# print(env.step((2, 2)))