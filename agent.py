
import numpy as np
import torch
from torch.optim import RMSprop
import copy
from rnn_agent import RNNAgent
from globalq import GlobalCritic, GlobalActor
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

        self.globalQ_critic = GlobalCritic(args)
        self.globalQ_actor = GlobalActor(args)
        self.target_globalQ_critic = copy.deepcopy(self.globalQ_critic)

        self.replay_memory = ReplayBuffer(args['buffer_size'], self.batch_size)

        self.localQ_params = list(self.mac.parameters())
        self.globalQ_critic_params = list(self.globalQ_critic.parameters())
        self.globalQ_actor_params = list(self.globalQ_actor.parameters())
        self.globalQ_params = self.globalQ_critic_params + self.globalQ_actor_params
        self.params = self.localQ_params + self.globalQ_params

        self.localQ_optimizer = RMSprop(params=self.localQ_params, lr=self.lr, alpha=args['optim_alpha'],
                                        eps=args['optim_eps'])
        self.globalQ_critic_optimizer = RMSprop(params=self.globalQ_critic_params, lr=self.global_lr, alpha=args['optim_alpha'],
                                         eps=args['optim_eps'])
        self.globalQ_actor_optimizer = RMSprop(params=self.globalQ_actor_params, lr=self.global_lr, alpha=args['optim_alpha'],
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
        state_var = torch.FloatTensor(state).unsqueeze(0)
        q_out, h = self.mac(state_var, h, 1, True)
        print('state', state, 'agent1', q_out[0].tolist(), 'agent2', q_out[1].tolist())

        probs_dist, global_a = self.globalQ_actor(state_var)
        max_action = probs_dist.sample().long().squeeze().tolist()
        global_a = global_a.squeeze().tolist()
        print('select', global_a, max_action)

        # return max_action

        # Action of global
        for i in range(self.n_agents):
            if train and (
                    step < self.batch_size * self.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
                action = np.random.choice(self.action_dim)
                act_n.append(action)
            else:
                act_n.append(max_action[i])

        return np.array(act_n, dtype=np.int32)


    def train_agents(self, state, action, reward, state_n, done, episode_num):

        # if np.random.rand() < self.epsilon:
        self.replay_memory.add_to_memory((state, action, reward, state_n, done))
        # else:
        #     self.replay_memory.add_to_memory((state, [0, 0], [8.0], state_n, done))

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
            global_q = self.globalQ_critic(batch_state, batch_actions)
            print(global_q[0], batch_actions[0])
            chosen_g_action_qvals = torch.gather(global_q, dim=2, index=batch_actions.unsqueeze(2)).squeeze(2)  # Remove the last dim

            default_actions = torch.ones(batch_actions.size(), dtype=torch.long) * 2
            default_g_action_qvals = torch.gather(global_q, dim=2, index=default_actions.unsqueeze(2)).squeeze(2)

            #TODO: Wrong!!! next action set is needed!!!! batch_actions --> batch_next_actions (but thi time we test on 1 step game)
            target_global_q = self.target_globalQ_critic(batch_state_n, batch_actions)

            cur_max_actions = global_q.max(dim=2, keepdim=True)[1]
            target_g_max_qvals = torch.gather(target_global_q, dim=2, index=cur_max_actions).squeeze(2)

            # Calculate 1-step Q-Learning targets
            targets_g = batch_rewards + self.gamma * (1 - batch_terminated.unsqueeze(1)) * target_g_max_qvals

            # Td-error
            td_error_g = (chosen_g_action_qvals - targets_g.detach())

            # 0-out the targets that came from padded data
            masked_td_error_g = td_error_g

            # Normal L2 loss, take mean over actual data
            loss_g_critic = (masked_td_error_g ** 2).mean()

            dist, global_a_val = self.globalQ_actor(batch_state)
            # chosen_global_a_val = torch.gather(global_a_val, dim=2, index=batch_actions.unsqueeze(2)).squeeze(2)
            loss_g_actor_n = -1 * td_error_g * dist.log_prob(batch_actions)
            # print('--------->', chosen_g_action_qvals, dist.log_prob(batch_actions))
            loss_g_actor = loss_g_actor_n.mean()
            # print(loss_g_actor, loss_g_critic)

            loss_g = loss_g_actor + 0.1 * loss_g_critic
            print('global loss', loss_g)


            self.globalQ_optimizer.zero_grad()
            loss_g.backward()
            grad_norm_global = torch.nn.utils.clip_grad_norm_(self.globalQ_params, self.grad_norm_clip)
            self.globalQ_optimizer.step()

            # self.globalQ_actor_optimizer.zero_grad()
            # self.globalQ_critic_optimizer.zero_grad()
            # loss_g_actor.backward()
            # loss_g_critic.backward()
            # grad_norm_critic = torch.nn.utils.clip_grad_norm_(self.globalQ_critic_params, self.grad_norm_clip)
            # gard_norm_actor = torch.nn.utils.clip_grad_norm(self.globalQ_actor_params, self.grad_norm_clip)
            # self.globalQ_actor_optimizer.step()
            # self.globalQ_critic_optimizer.step()


            if episode_num > self.target_update_interval:
                # for each local Q function
                # Calculate estimated Q-Values
                h = self.mac.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
                mac_out, h = self.mac.forward(batch_state, h, self.batch_size)
                mac_out = mac_out.view(self.batch_size, self.n_agents, -1)
                # print(mac_out[0])

                # Pick the Q-Values for the actions taken by each agent
                chosen_action_qvals = torch.gather(mac_out, dim=2, index=batch_actions.unsqueeze(2)).squeeze(2)  # Remove the last dim
                default_actions = torch.ones(batch_actions.size(), dtype=torch.long) * 2
                default_action_qvals = torch.gather(mac_out, dim=2, index=default_actions.unsqueeze(2)).squeeze(2)

                h = self.target_mac.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
                target_mac_out, h = self.target_mac.forward(batch_state_n, h, self.batch_size)
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
                # diff_individual = chosen_action_qvals - default_action_qvals
                # idx_pos = torch.sign(diff_individual) > 0
                # idx_neg = torch.sign(diff_rewards[idx_pos]) < 0
                # batch_agents = torch.zeros(batch_actions.size())
                # batch_agents[:, 1] = 1
                # print(diff_individual)
                # if len(diff_rewards[idx_pos][idx_neg]) > 0:
                #     print('!!!!!!!!!!', diff_individual[idx_pos][idx_neg])
                #     print('??????????', diff_rewards[idx_pos][idx_neg])
                #     print('action', batch_actions[idx_pos][idx_neg])
                #     print('whowho', batch_agents[idx_pos][idx_neg])

                # Td-error
                td_error = (chosen_action_qvals - targets.detach())
                print('chosen local', chosen_action_qvals[0], batch_actions[0])

                # Normal L2 loss, take mean over actual data
                loss = (td_error ** 2).mean() # - 0.5 * torch.tensor(diff_rewards.tolist()) * diff_individual).mean()
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
        self.target_globalQ_critic.load_state_dict(self.globalQ_critic.state_dict())
        print("Updated target network")
