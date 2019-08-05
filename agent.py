
import numpy as np
import torch
from torch.optim import RMSprop
import copy
from rnn_agent import RNNAgent
from globalq import GlobalQ
from replay_buffer import ReplayBuffer


class PotentialAgent():
    def __init__(self, args):
        self.epsilon = 1.0
        self.epsilon_dec = 2.0 / args['training_step']
        self.epsilon_min = 0.1
        self.gamma = args['discount_factor']
        self.lr = args['lr']
        self.global_lr = args['global_lr']
        self.grad_norm_clip = args['grad_norm_clip']
        self.target_update_interval = args['target_update_interval']
        self.last_target_update_episode = 0

        self.n_agents = args['n_agents']
        self.action_dim = args['action_dim']
        self.batch_size = args['batch_size']
        self.pre_train_step = args['pre_train_step']

        self.mac = RNNAgent(args)
        self.target_mac = copy.deepcopy(self.mac)

        self.globalQ = GlobalQ(args)
        self.target_globalQ = copy.deepcopy(self.globalQ)

        self.replay_memory = ReplayBuffer(args['buffer_size'], self.batch_size)

        self.localQ_params = list(self.mac.parameters())
        self.globalQ_params = list(self.globalQ.parameters())
        self.params = self.localQ_params + self.globalQ_params

        self.localQ_optimizer = RMSprop(params=self.localQ_params, lr=self.lr, alpha=args['optim_alpha'],
                                        eps=args['optim_eps'])
        self.globalQ_optimizer = RMSprop(params=self.globalQ_params, lr=self.global_lr, alpha=args['optim_alpha'],
                                         eps=args['optim_eps'])

    def get_action(self, state, step, obs, train=True):
        act_n = []
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_min)
        print(self.epsilon)

        h = self.mac.init_hidden().unsqueeze(0).expand(1, self.n_agents, -1)
        action_dummy = np.zeros(self.n_agents).astype(np.int)
        batch = [(state, action_dummy)]
        batch = np.array(batch)
        # obs_var = torch.FloatTensor(obs).unsqueeze(0)
        q_out, h = self.mac(batch, h, 1, True)
        print('agent1', q_out[0].tolist(), 'agent2', q_out[1].tolist())
        max_q_out = torch.argmax(q_out, dim=-1)
        # act_n.append(torch.argmax(q_out).item())

        # Action of predator
        for i in range(self.n_agents):
            if train and (
                    step < self.batch_size * self.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
                action = np.random.choice(self.action_dim)
                act_n.append(action)
            else:
                act_n.append(max_q_out[i])

        return np.array(act_n, dtype=np.int32)


    def train_agents(self, state, action, reward, state_n, done, episode_num):

        if np.random.rand() < self.epsilon:
            self.replay_memory.add_to_memory((state, action, reward, state_n, done))
        else:
            self.replay_memory.add_to_memory((state, [0, 0], [8.0], state_n, done))

        if episode_num > self.batch_size:
            batch = self.replay_memory.sample_from_memory()
            batch = np.array(batch)
            bs = self.batch_size
            batch_state = torch.from_numpy(np.array(list(batch[:, 0]))).type(torch.float32) # state
            batch_actions = torch.from_numpy(np.array(list(batch[:, 1]))).type(torch.long) # action
            batch_rewards = torch.from_numpy(np.array(list(batch[:, 2]))).type(torch.float32) # reward
            batch_state_n = torch.from_numpy(np.array(list(batch[:, 3]))).type(torch.float32)  # next state
            batch_terminated = torch.from_numpy(np.array(list(batch[:, 4]))).type(torch.float32) # done

            # Optimize Global Q
            hidden_states = self.globalQ.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
            global_q, hidden_states = self.globalQ(batch, hidden_states)
            print(global_q[0], batch_actions[0])
            chosen_g_action_qvals = torch.gather(global_q, dim=2, index=batch_actions.unsqueeze(2)).squeeze(2)  # Remove the last dim

            default_actions = torch.ones(batch_actions.size(), dtype=torch.long) * 2
            default_g_action_qvals = torch.gather(global_q, dim=2, index=default_actions.unsqueeze(2)).squeeze(2)

            target_hidden_states = self.globalQ.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
            target_global_q, target_hidden_states = self.target_globalQ(batch, target_hidden_states)

            cur_max_actions = global_q.max(dim=2, keepdim=True)[1]
            target_g_max_qvals = torch.gather(target_global_q, dim=2, index=cur_max_actions).squeeze(2)

            # Calculate 1-step Q-Learning targets
            targets_g = batch_rewards + self.gamma * (1 - batch_terminated.unsqueeze(1)) * target_g_max_qvals

            # Td-error
            td_error_g = (chosen_g_action_qvals - targets_g.detach())


            # 0-out the targets that came from padded data
            masked_td_error_g = td_error_g

            # Normal L2 loss, take mean over actual data
            loss_g = (masked_td_error_g ** 2).mean()
            self.globalQ_optimizer.zero_grad()
            loss_g.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.globalQ_params, self.grad_norm_clip)
            self.globalQ_optimizer.step()

            if episode_num > self.target_update_interval:
                # for each local Q function
                # Calculate estimated Q-Values
                h = self.mac.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
                mac_out, h = self.mac.forward(batch, h, self.batch_size)
                mac_out = mac_out.view(self.batch_size, self.n_agents, -1)
                print(mac_out[0])

                # Pick the Q-Values for the actions taken by each agent
                chosen_action_qvals = torch.gather(mac_out, dim=2, index=batch_actions.unsqueeze(2)).squeeze(2)  # Remove the last dim
                default_actions = torch.ones(batch_actions.size(), dtype=torch.long) * 2
                default_action_qvals = torch.gather(mac_out, dim=2, index=default_actions.unsqueeze(2)).squeeze(2)

                h = self.target_mac.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
                target_mac_out, h = self.target_mac.forward(batch, h, self.batch_size)
                target_mac_out = target_mac_out.view(self.batch_size, self.n_agents, -1)

                # Mask out unavailable actions

                # Max over target Q-Values
                # Get actions that maximise live Q (for double q-learning)
                cur_max_actions = mac_out.max(dim=2, keepdim=True)[1]
                target_max_qvals = torch.gather(target_mac_out, 2, cur_max_actions).squeeze(2)

                diff_rewards = chosen_g_action_qvals - default_g_action_qvals
                print('chosen', chosen_g_action_qvals[0], 'default', default_g_action_qvals[0])
                print('diff reward', diff_rewards[0])

                # Calculate 1-step Q-Learning targets
                targets = diff_rewards + self.gamma * (1 - batch_terminated.unsqueeze(1)) * target_max_qvals

                # # for debug value
                # kkkk = chosen_action_qvals - default_action_qvals
                # idx_pos = torch.sign(kkkk) > 0
                # idx_neg = torch.sign(diff_rewards[idx_pos]) < 0
                # if len(diff_rewards[idx_pos][idx_neg]) > 0:
                #     print('!!!!!!!!!!', kkkk[idx_pos][idx_neg])
                #     print('??????????', diff_rewards[idx_pos][idx_neg])

                # Td-error
                td_error = (chosen_action_qvals - targets.detach())
                print('chosen local', chosen_action_qvals[0], batch_actions[0])

                # Normal L2 loss, take mean over actual data
                loss = (td_error ** 2).mean()
                print('loss', loss.item())

                # Optimise
                self.localQ_optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.localQ_params, self.grad_norm_clip)
                self.localQ_optimizer.step()

            if (episode_num - self.last_target_update_episode) / self.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = episode_num


    def _update_targets(self):
        self.target_mac.load_state_dict(self.mac.state_dict())
        self.target_globalQ.load_state_dict(self.globalQ.state_dict())
        print("Updated target network")
