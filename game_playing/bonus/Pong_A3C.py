
# coding: utf-8

# In[1]:

import argparse
import os
import sys
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from gym import wrappers
import time
import gym
import numpy as np
from gym.spaces.box import Box
import cv2


# In[2]:

parser = argparse.ArgumentParser(description='A3C')

parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')

parser.add_argument('--num-processes', type=int, default=4, metavar='NP',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=100000, metavar='M',
                    help='maximun length of an episode (default: 100000)')
parser.add_argument('--env-name', default='Pong-v0', metavar='ENV',
                    help='environment to train on (default: Pong-v0)')


# In[3]:

class ActorCritic(torch.nn.Module):
    
    def __init__(self, num_inputs, action_space):
        
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        # LSTM
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        
        # actor-critic
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        self.train()
        
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        
        x = x.view(-1, 32 * 3 * 3) 
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


# In[4]:

class SharedAdam(optim.Adam):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        
        super(SharedAdam, self).__init__(params, lr, betas, eps)
        
        # init to 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
    
    # share adam's param
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
    
    # update weight
    def step(self):
        loss = None
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                    
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # update first moment estimate & second moment estimate
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * np.sqrt(bias_correction2) / bias_correction1
                
                # inplce mode of addcdiv
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return loss


# In[5]:

class AtariRescale42x42(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation):
        return _process_frame42(observation) 

class NormalizedEnv(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha +             observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha +             observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        ret = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return np.expand_dims(ret, axis=0)


# In[6]:

def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def write_score(episode, score):
    with open("/home/leoqaz12/GoogleDrive/HW3_model/Pong_A3C_score1.csv", "a") as myfile:
        myfile.write("%d, %4f \n" % (episode, score))


def weights_init(m):
    classname = m.__class__.__name__
    #initialization
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None: return
        shared_param._grad = param.grad


# In[7]:

def actor(rank, args, shared_model, optimizer):

    env = create_atari_env(args.env_name)
    

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.train()
    
    # init
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    
    while True:
        episode_length += 1
        
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        
        # LSTM's param
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(args.num_steps):
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            

            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            
            # gym env step
            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)
            
            if done:
                episode_length = 0
                state = env.reset()
            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                break
                
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
            
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)
            
            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i+1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss += -(log_probs[i] * Variable(gae) + 0.01 * entropies[i])
            
        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        
        ensure_shared_grads(model, shared_model)
        
        # update
        optimizer.step()


# In[8]:

def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    return frame

def monitor(rank, args, shared_model):
    
    env = create_atari_env(args.env_name)
    env = wrappers.Monitor(env, './video/pong-a3c', video_callable=lambda count: count % 30 == 0, force=True)
    
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    # eval mode
    model.eval()
    
    # init
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    episode_length = 0
    done = True
    start_time = time.time()
    episodeNUM = 0
    # rewardNUM=0
    hist = -21
    
    while True:
        #env.render()
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True) # lstm's param
            hx = Variable(torch.zeros(1, 256), volatile=True) # lstm's param
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)
            
        value, logit, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        
        state, reward, done, _ = env.step(action[0][0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward
        
        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            # reset
            
            episode_length = 0
            state = env.reset()
            time.sleep(60)
            episodeNUM += 1
            # rewardNUM += reward_sum
            # if episodeNUM % 30 ==0:
            hist = hist*0.8 + reward_sum*0.2
            write_score(episodeNUM*30, hist)
            # rewardNUM = 0
            print("ep:"+str(episodeNUM)+"=="+str(hist)+"==")
            reward_sum = 0
                
            
        state = torch.from_numpy(state)


# In[9]:

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    env = create_atari_env(args.env_name)
    # Critic
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()
    # optimizer, adam with shared statistics
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []

    p = mp.Process(target=monitor, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=actor, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# In[10]:

#%tb

