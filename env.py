import gym
import numpy as np

# from QTRAN code
class MultiAgentSimpleEnv2(gym.Env):  # Matrix game
    def __init__(self):
        self.state_set = np.eye(3) # state 1, 2A, 2B
        self.state = 0
        self.action_dim = 3
        self.state_dim = 3

        self.payoff_A = np.array([[4., 4., 4.], [4., 4., 4.], [4., 4., 4.]])
        self.payoff_B = np.array([[8., 5., 2.], [5., 2., 1.], [2., 1., 0.]])
    def reset(self):
        self.state = 0
        state = self.state_set[self.state]

        return state

    def step(self, action):
        info = {'n': []}
        reward = []
        done = []
        if self.state == 0:
            if action[0] == 1 or action[0] == 2:
                next_state = 1
            elif action[0] == 0:
                next_state = 2
            reward.append(0)
            done.append(False)

        elif self.state == 1:
            next_state = 0
            reward.append(self.payoff_A[action[0]][action[1]])
            done.append(True)

        elif self.state == 2:
            next_state = 0
            reward.append(self.payoff_B[action[0]][action[1]])
            done.append(True)

        self.state = next_state
        next_state_out = self.state_set[next_state]


        return next_state_out, reward, done, info

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