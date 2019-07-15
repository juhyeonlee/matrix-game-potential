import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class RNNAgent(nn.Module):
    def __init__(self, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args['n_agents']
        self.n_actions = args['action_dim']
        self.input_shape = self._get_input_shape(args)

        self.fc1 = nn.Linear(self.input_shape, args['rnn_hidden_dim'])
        self.rnn = nn.GRUCell(args['rnn_hidden_dim'], args['rnn_hidden_dim'])
        self.fc2 = nn.Linear(args['rnn_hidden_dim'], args['action_dim'])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args['rnn_hidden_dim']).zero_()

    def forward(self, inputs, hidden_state, bs):
        x = F.relu(self.fc1(self._build_inputs(inputs, bs).view(-1, self.input_shape)))
        h_in = hidden_state.reshape(-1, self.args['rnn_hidden_dim'])
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

    def _build_inputs(self, batch, bs):
        # max_t = batch.max_seq_length if t is None else 1
        # ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        if bs == 1:
            batch_state = batch
        else:
            batch_state = torch.from_numpy(np.array(list(batch[:, 0])))

        # observation
        inputs.append(batch_state.unsqueeze(1).repeat(1, self.n_agents, 1).type(torch.float32))

        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, args):
        # observation
        input_shape = args['state_dim']
        # agent id
        input_shape += self.n_agents
        return input_shape
