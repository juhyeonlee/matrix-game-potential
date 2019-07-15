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
        input_shape = self._get_input_shape(args)
        self.rnn_hidden_dim = 128

        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        # self.fc2 = nn.Linear(128, 128)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc3 = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def forward(self, batch, hidden_state):
        inputs = self._build_inputs(batch)
        x = F.relu(self.fc1(inputs))
        # print(x.size(), hidden_state.size())
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        x_in = x.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x_in, h_in)
        q = self.fc3(h)
        q = q.reshape(self.batch_size, self.n_agents, -1)
        h = h.reshape(self.batch_size, self.n_agents, -1)
        return q, h

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def _build_inputs(self, batch):
        bs = self.batch_size
        # max_t = batch.max_seq_length if t is None else 1
        # ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        batch_state = torch.from_numpy(np.array(list(batch[:, 0])))
        batch_action = torch.from_numpy(np.array(list(batch[:, 1])))
        # state
        inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))

        # observation
        inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))

        # actions (masked out by agent)
        batch_action_onehot = torch.from_numpy(np.eye(self.n_actions)[batch_action])
        actions = batch_action_onehot.view(bs, 1, -1).repeat(1, self.n_agents, 1)
        actions = actions.type(torch.float32)
        agent_mask = (1 - torch.eye(self.n_agents))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0))

        # # last actions
        # if t == 0:
        #     inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        # elif isinstance(t, int):
        #     inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        # else:
        #     last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
        #     last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        #     inputs.append(last_actions)

        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, args):
        # state
        input_shape = args['state_dim']
        # observation
        input_shape += args['state_dim']
        # actions and last actions
        input_shape += args["action_dim"] * self.n_agents
        # agent id
        input_shape += self.n_agents
        return input_shape
