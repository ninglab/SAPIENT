import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import *
from sum_tree import SumTree
from RL.env_multi_choice_question import MultiChoiceRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from multi_interest import GraphEncoder
import time
import warnings
import ipdb
from dqn import DQN

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand'))
        
class Agent(object):
    def __init__(self, device, memory, state_size, action_size, hidden_size, gcn_net, learning_rate, l2_norm, PADDING_ID, EPS_START = 0.9, EPS_END = 0.1, EPS_DECAY = 0.0001, tau=0.01):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.gcn_net = gcn_net
        self.policy_net = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(device)
        self.policy = Policy(state_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(chain(self.policy_net.parameters(),self.gcn_net.parameters(),self.policy.parameters()), lr=learning_rate, weight_decay = l2_norm)
        self.optimizer_q_network = optim.Adam(chain(self.policy_net.parameters(),self.gcn_net.parameters()), lr=learning_rate, weight_decay = l2_norm)
        self.optimizer_policy_network = optim.Adam(chain(self.policy.parameters(),self.gcn_net.parameters()), lr=learning_rate, weight_decay = l2_norm)
        self.memory = memory
        self.loss_func = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.PADDING_ID = PADDING_ID
        self.tau = tau

    def select_feature(self, state, cand1, action_space, is_test=False, is_last_turn=False):
        feature, item = cand1
        cand2 = feature
        state_emb = self.gcn_net([state])
        cand = torch.LongTensor([cand2]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                actions_value = self.policy_net(state_emb, cand_emb)
                action = cand[0][actions_value.argmax().item()]
                sorted_actions = cand[0][actions_value.sort(1, True)[1].tolist()]
                return action, sorted_actions.tolist(),state_emb
        else:
            shuffled_cand = action_space[0]
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long), shuffled_cand,state_emb
            
    def select_item(self, state, cand1, action_space, is_test=False, is_last_turn=False):
        feature, item = cand1
        cand2 = item
        state_emb = self.gcn_net([state])
        cand = torch.LongTensor([cand2]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            if is_test and (len(action_space[1]) <= 20 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1],state_emb
            with torch.no_grad():
                actions_value = self.policy_net(state_emb, cand_emb)
                action = cand[0][actions_value.argmax().item()]
                sorted_actions = cand[0][actions_value.sort(1, True)[1].tolist()]
                return action, sorted_actions.tolist(),state_emb
        else:
            shuffled_cand = action_space[1]
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long), shuffled_cand,state_emb
    
    def select_action(self, state, cand1, action_space, env, is_test=False, is_last_turn=False):
        state_emb = self.gcn_net([state])
        feature, item = cand1
        if is_last_turn:
            cand2 = item
        else:
            if (not feature) or (not item):
                cand2 = item+feature
            else:
                selection = Categorical(self.policy(state_emb).softmax(dim=-1).squeeze()).sample()
                if selection == 0:
                    cand2 = feature
                else:
                    cand2 = item
        cand = torch.LongTensor([cand2]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            if is_test and (len(action_space[1]) <= 20 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1],state_emb
            with torch.no_grad():
                actions_value = self.policy_net(state_emb, cand_emb)
                action = cand[0][actions_value.argmax().item()]
                sorted_actions = cand[0][actions_value.sort(1, True)[1].tolist()]
                return action, sorted_actions.tolist(),state_emb
        else:
            if is_last_turn:
                shuffled_cand = action_space[1]
            else:
                shuffled_cand = action_space[0]+action_space[1]
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long), shuffled_cand,state_emb

    
    def update_target_model(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))
            
            
    def optimize_q_network(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        self.update_target_model()
        
        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_emb_batch = self.gcn_net(list(batch.state))
        action_batch = torch.stack(batch.action).detach().cpu()
        action_batch = torch.LongTensor(np.array(action_batch).astype(int).reshape(-1, 1)).to(self.device) #[N*1]
        action_emb_batch = self.gcn_net.embedding(action_batch)
        reward_batch = torch.stack(batch.reward).detach().cpu()
        reward_batch = torch.FloatTensor(np.array(reward_batch).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = []
        n_cands = []
        for s, c in zip(batch.next_state, batch.next_cand):
            if s is not None:
                n_states.append(s)
                n_cands.append(c[0]+c[1])
        next_state_emb_batch = self.gcn_net(n_states)
        next_cand_batch = self.padding(n_cands)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)
        q_eval = self.policy_net(state_emb_batch, action_emb_batch, choose_action=False)  
        best_actions = torch.gather(input=next_cand_batch, dim=1, index=self.policy_net(next_state_emb_batch,next_cand_emb_batch,Op=True).argmax(dim=1).view(len(n_states),1).to(self.device))
        best_actions_emb = self.gcn_net.embedding(best_actions)
        q_target = torch.zeros((BATCH_SIZE,1), device=self.device)
        q_target[non_final_mask] = self.target_net(next_state_emb_batch,best_actions_emb,choose_action=False).detach()
        q_target = reward_batch + GAMMA * q_target
        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)
        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()
        self.optimizer_q_network.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_q_network.step()
        return loss.data
        
    def optimize_policy_network(self, BATCH_SIZE, total_buffer):
        if len(total_buffer) < 1:
            return
        else:
            BATCH_SIZE = 1
        total_loss = None
        for b in range(BATCH_SIZE):
            buffer = total_buffer.sample()
            keys = buffer.keys()
            sorted_traj_idx = sorted(buffer.keys(), key = lambda x: buffer[x][1])
            sorted_traj_idx = sorted(sorted_traj_idx, key = lambda x: -buffer[x][2])
            #save memory
            short_traj_idx = []
            for x in range(len(sorted_traj_idx)):
                if x%5==0:
                    short_traj_idx.append(sorted_traj_idx[x])
            sorted_traj_idx = short_traj_idx
            traj_likelihood = {}
            idx_best = sorted_traj_idx[0]
            best_traj_length = buffer[idx_best][1]
            for idx in sorted_traj_idx:
                traj_length = buffer[idx][1]
                state_embs = self.gcn_net([x[0] for x in buffer[idx][0]][:best_traj_length])
                action_types = torch.LongTensor([x[1] for x in buffer[idx][0]][:best_traj_length]).to(self.device)
                traj_likelihood[idx] = torch.exp(torch.sum(torch.log(torch.gather(self.policy(state_embs).softmax(dim=-1), 1, action_types.unsqueeze(1)))))
            loss = 1.0
            for i in range(len(sorted_traj_idx)):
                idx = sorted_traj_idx[i]
                loss *= traj_likelihood[idx]
                loss /= torch.sum(torch.stack([traj_likelihood[sorted_traj_idx[j]] for j in range(i,len(sorted_traj_idx))]))
            if total_loss is not None:
                total_loss += -torch.log(loss)
            else:
                total_loss = -torch.log(loss)
        total_loss /= BATCH_SIZE
        self.optimizer_policy_network.zero_grad()
        total_loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_policy_network.step()
        return total_loss.data

    def optimize_model(self, BATCH_SIZE, GAMMA, env):
        if len(self.memory) < BATCH_SIZE:
            return
        
        self.update_target_model()
        
        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_emb_batch = self.gcn_net(list(batch.state))
        action_batch = torch.stack(batch.action).detach().cpu()
        action_batch = torch.LongTensor(np.array(action_batch).astype(int).reshape(-1, 1)).to(self.device) #[N*1]
        action_emb_batch = self.gcn_net.embedding(action_batch)
        reward_batch = torch.stack(batch.reward).detach().cpu()
        reward_batch = torch.FloatTensor(np.array(reward_batch).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = []
        n_cands = []
        for s, c in zip(batch.next_state, batch.next_cand):
            if s is not None:
                n_states.append(s)
                n_cands.append(c[0]+c[1])
        next_state_emb_batch = self.gcn_net(n_states)
        next_cand_batch = self.padding(n_cands)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)
        q_eval = self.policy_net(state_emb_batch, action_emb_batch, choose_action=False)  
        best_actions = torch.gather(input=next_cand_batch, dim=1, index=self.policy_net(next_state_emb_batch,next_cand_emb_batch,Op=True).argmax(dim=1).view(len(n_states),1).to(self.device))
        best_actions_emb = self.gcn_net.embedding(best_actions)
        q_target = torch.zeros((BATCH_SIZE,1), device=self.device)
        q_target[non_final_mask] = self.target_net(next_state_emb_batch,best_actions_emb,choose_action=False).detach()
        q_target = reward_batch + GAMMA * q_target
        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)
        action_type = (action_batch<(env.user_length + env.item_length)).long().squeeze()
        policy_logits = self.policy(state_emb_batch)
        loss_policy = self.cross_entropy_loss(policy_logits, action_type)
        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean() + loss_policy
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.data
    
    def save_model(self, data_name, filename, epoch_user):
        model_file = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}.tar'.format(epoch_user)
        if not os.path.isdir(TMP_DIR[data_name] + '/RL-agent/'):
            os.makedirs(TMP_DIR[data_name] + '/RL-agent/')
        torch.save({
            'epoch':epoch_user,
            'policy_model_state_dict':self.policy_net.state_dict(),
            'target_model_state_dict':self.target_net.state_dict(),
            'gcn_model_state_dict':self.gcn_net.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()
            }, model_file)
        print('RL policy model saved at {}'.format(model_file))
        
        
    def load_model(self, data_name, filename, epoch_user):
        model_file = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}.tar'.format(epoch_user)
        checkpoint = torch.load(model_file)
        self.policy_net.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_model_state_dict"])
        self.gcn_net.load_state_dict(checkpoint["gcn_model_state_dict"])
        epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return epoch
    
    def padding(self, cand):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.PADDING_ID
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)

class Policy(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, X):
        all_emb = torch.cat(X, dim=-1)
        logits = self.fc2(F.tanh(self.fc1(all_emb.squeeze())))
        return logits
