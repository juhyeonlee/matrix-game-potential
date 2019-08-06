import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GlobalQ(nn.Module):
    def __init__(self, args):
        super(GlobalQ, self).__init__()

        self.args = args
        self.n_actions = args['action_dim']
        self.n_agents = args['n_agents']
        self.batch_size = args['batch_size']
        self.gamma = args['discount_factor']
        self.global_lr = args['global_lr']
        self.q_table = np.zeros((self.n_actions, self.n_actions), dtype=np.float)


    def forward(self, batch):
        q = []
        for idx in range(self.batch_size):
            a = batch[idx]
            q.append(self.q_table[a[0]][a[1]])
        q = torch.tensor(q)
        return q

    def learn(self, batch):
        batch_action = torch.from_numpy(np.array(list(batch[:, 1])))
        batch_reward = torch.from_numpy(np.array(list(batch[:, 2])))

        for idx in range(self.batch_size):
            a = batch_action[idx]
            new_q = batch_reward[idx] #+ self.gamma * max(self.q_table[next_state])
            self.q_table[a[0]][a[1]] += self.global_lr * (new_q - self.q_table[a[0]][a[1]])
    # def init_hidden(self):
    #     return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    #
    # def _build_inputs(self, batch):
    #     bs = self.batch_size
    #     # max_t = batch.max_seq_length if t is None else 1
    #     # ts = slice(None) if t is None else slice(t, t+1)
    #     inputs = []
    #     batch_state = torch.from_numpy(np.array(list(batch[:, 0])))
    #     batch_action = torch.from_numpy(np.array(list(batch[:, 1])))
    #     # state
    #     inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))
    #
    #     # observation
    #     inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))
    #
    #     # actions (masked out by agent)
    #     batch_action_onehot = torch.from_numpy(np.eye(self.n_actions)[batch_action])
    #     actions = batch_action_onehot.view(bs, 1, -1).repeat(1, self.n_agents, 1)
    #     actions = actions.type(torch.float32)
    #     agent_mask = (1 - torch.eye(self.n_agents))
    #     agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
    #     inputs.append(actions * agent_mask.unsqueeze(0))
    #
    #     # # last actions
    #     # if t == 0:
    #     #     inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
    #     # elif isinstance(t, int):
    #     #     inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
    #     # else:
    #     #     last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
    #     #     last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
    #     #     inputs.append(last_actions)
    #
    #     inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))
    #
    #     inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
    #     return inputs
    #
    # def _get_input_shape(self, args):
    #     # state
    #     input_shape = args['state_dim']
    #     # observation
    #     input_shape += args['state_dim']
    #     # actions and last actions
    #     input_shape += args["action_dim"] * self.n_agents
    #     # agent id
    #     input_shape += self.n_agents
    #     return input_shape
