import numpy as np
import torch
from env import MultiAgentSimpleEnv2
from agent import PotentialAgent

torch.manual_seed(123)
np.random.seed(123)


env = MultiAgentSimpleEnv2()

args = {}
args['training_step'] = 5000
args['max_step'] = 200
args['pre_train_step'] = 10
args['n_agents'] = 2
args['action_dim'] = env.action_dim
args['state_dim'] = env.state_dim
args['discount_factor'] = 0.99
args['buffer_size'] = 50000
args['batch_size'] = 32
args['lr'] = 0.0005
args['global_lr'] = 0.0005
args['optim_alpha'] = 0.99
args['optim_eps'] = 0.00001
args['rnn_hidden_dim'] = 64
args['grad_norm_clip'] = 10
args['target_update_interval'] = 200
agent = PotentialAgent(args)


step = 0
episode = 0
while step < args['training_step']:
    episode += 1
    ep_step = 0
    obs = env.reset()
    state = obs
    total_reward = 0

    while True:
        step += 1
        ep_step += 1
        action = agent.get_action(state, step, obs)
        obs_n, reward, done, info = env.step(action)
        state_n = obs_n

        done_single = sum(done) > 0
        if ep_step >= args['max_step']:
            done_single = True
        agent.train_agents(state, action, reward, state_n, done_single, episode)

        obs = obs_n
        state = state_n
        total_reward += np.sum(reward) * (args['discount_factor'] ** (ep_step - 1))
        # if step % 100 ==0:
        #    print(step, agent.globalQ.)
        if done_single or ep_step >= args['max_step']:
            print("[train_ep %d]" % (episode), "\tstep:", step, "\tep_step:", ep_step, "\treward", total_reward,
                  "\taction", action)
            break
    #
    # if episode % eval_step == 0:
    #     self.test(episode)
    #
    # self._eval.summarize()



