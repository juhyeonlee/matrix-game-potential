import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GlobalCritic(nn.Module):
    def __init__(self, args):
        super(GlobalCritic, self).__init__()

        self.args = args
        self.n_actions = args['action_dim']
        self.n_agents = args['n_agents']
        self.batch_size = args['batch_size']
        input_shape = self._get_input_shape(args)
        self.hidden_dim = 128
        # self.rnn_hidden_dim = 128

        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(self, batch_state, batch_action):
        inputs = self._build_inputs(batch_state, batch_action)
        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        # x_in = x.reshape(-1, self.rnn_hidden_dim)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        q = q.reshape(self.batch_size, self.n_agents, -1)
        # h = h.reshape(bs, self.n_agents, -1)
        return q #, h
    #
    # def init_hidden(self):
    #     return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def _build_inputs(self, batch_state, batch_action):
        bs = self.batch_size
        # max_t = batch.max_seq_length if t is None else 1
        # ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))

        # observation
        inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))

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


class GlobalActor(nn.Module):
    def __init__(self, args):
        super(GlobalActor, self).__init__()

        self.args = args
        self.n_actions = args['action_dim']
        self.n_agents = args['n_agents']
        self.batch_size = args['batch_size']
        self.hidden_dim = 128
        input_shape = self._get_input_shape(args)

        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(self, batch_state):

        x = F.relu(self.fc1(self._build_input(batch_state)))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        softmax_output = F.softmax(output, dim=-1)
        probs_dist = torch.distributions.Categorical(softmax_output)
        return probs_dist, softmax_output

    def _build_input(self, batch_state):
        inputs = []
        bs = batch_state.size()[0]

        # state
        inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))

        # observation
        inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))

        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, args):
        # state
        input_shape = args['state_dim']
        # observation
        input_shape += args['state_dim']
        # agent id
        input_shape += self.n_agents
        return input_shape

