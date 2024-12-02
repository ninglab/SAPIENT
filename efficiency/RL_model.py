import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
from utils import *
from agent import Agent
#TODO select env
from RL.env_multi_choice_question import MultiChoiceRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from multi_interest import GraphEncoder
import time
import warnings
from construct_graph import get_graph
from memory import *
from node import Node
import copy



pid = os.getpid()
print('pid : ',pid)
warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM_STAR: MultiChoiceRecommendEnv,
    YELP_STAR: MultiChoiceRecommendEnv,
    BOOK:MultiChoiceRecommendEnv,
    MOVIE:MultiChoiceRecommendEnv
    }
FeatureDict = {
    LAST_FM_STAR: 'feature',
    YELP_STAR: 'feature',
    BOOK:'feature',
    MOVIE:'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand'))

def train(args, kg, dataset, filename):
    
    SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best, reward_best=0.,0.,0.,15.,0.,0.
    env = EnvDict[args.data_name](kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn, cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                       attr_num=args.attr_num, mode='train', entropy_way=args.entropy_method, fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    G=get_graph(env.user_length,env.item_length,env.feature_length,args.data_name)
    memory = ReplayMemoryPER(args.memory_size)
    embed = torch.FloatTensor(np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1,env.ui_embeds.shape[1]))), axis=0))
    gcn_net = GraphEncoder(graph=G,device=args.device, entity=env.user_length+env.item_length+env.feature_length+1, emb_size=embed.size(1), kg=kg, embeddings=embed, \
        fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden,u=env.user_length,v=env.item_length,f=env.feature_length).to(args.device)
    agent = Agent(device=args.device, memory=memory, state_size=args.hidden, action_size=embed.size(1), \
        hidden_size=args.hidden, gcn_net=gcn_net, learning_rate=args.learning_rate, l2_norm=args.l2_norm, PADDING_ID=embed.size(0)-1)
    if args.load_rl_epoch != 0 :
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        args.load_rl_epoch = agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.eval_num == 10:
        SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
        test_performance.append(SR15_mean)    
    user_length = env.user_length
    item_length = env.item_length
    total_buffer = TotalBuffer()
    buffer = {x:[[],15, 0] for x in range(args.num_rollouts)}
    for train_step in range(args.load_rl_epoch+1, args.max_steps+1):
        SR5, SR10, SR15, AvgT, Rank, total_reward = 0., 0., 0., 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        root = Node()
        root.is_root = True
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))

            if not args.fix_emb:
                if i_episode%args.num_rollouts==0:
                    state, cand, action_space = env.reset(agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
                    current_user_id = env.user_id
                    current_acc_feature = env.user_acc_feature
                    current_target_item = env.target_item
                    buffer = {x:[[],15, 0] for x in range(args.num_rollouts)}
                    root = Node()
                    root.is_root = True
                else:
                    state, cand, action_space = env.reset_by_user_id_and_state(current_user_id, current_acc_feature, current_target_item, agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
            else:
                if i_episode%args.num_rollouts==0:
                    state, cand, action_space = env.reset()
                    current_user_id = env.user_id
                    current_acc_feature = env.user_acc_feature
                    current_target_item = env.target_item
                    buffer = {x:[[],15, 0] for x in range(args.num_rollouts)}
                    root = Node()
                    root.is_root = True
                else:
                    state, cand, action_space = env.reset_by_user_id_and_state(current_user_id, current_acc_feature, current_target_item) 
            epi_reward = 0
            is_last_turn = False
            current_node = root
            count = 0
            done = False
            while not done:
                if count == 14:
                    is_last_turn = True
                #selection
                if current_node.ask and current_node.rec:
                    #pay attention when there is no feature to select
                    if is_last_turn or (not cand[0]):
                        selection = 1
                    else:
                        selection  = current_node.select(args.w)
                    if selection == 0:
                        current_node = current_node.ask
                        action, sorted_actions,state_emb = agent.select_feature(state, cand, action_space, is_last_turn=is_last_turn)
                    elif selection == 1:
                        current_node = current_node.rec
                        action, sorted_actions,state_emb = agent.select_item(state, cand, action_space, is_last_turn=is_last_turn)
                else: 
                    #expansion
                    ask_node = Node()
                    ask_node.action = 0
                    ask_node.parent = current_node
                    rec_node = Node()
                    rec_node.action = 1
                    rec_node.parent = current_node
                    current_node.ask = ask_node
                    current_node.rec = rec_node
                    #pay attention when there is no feature to select
                    if is_last_turn or (not cand[0]):
                        action, sorted_actions,state_emb = agent.select_item(state, cand, action_space, is_last_turn=is_last_turn)
                    else:
                        action, sorted_actions,state_emb = agent.select_action(state, cand, action_space, env, is_last_turn=is_last_turn)
                    if action >= env.user_length + env.item_length:
                        current_node = current_node.ask
                    else:
                        current_node = current_node.rec
                #simulation
                if not args.fix_emb:
                    next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions, agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
                else:
                    next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions)
                epi_reward += reward
                current_node.r = reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                if done:
                    next_state = None
                agent.memory.push(state, action, next_state, reward, next_cand)
                action_type = int(action<(env.user_length + env.item_length))
                temp = (state, action_type)
                buffer[i_episode%args.num_rollouts][0].append(temp)
                if i_episode%5==0:
                    newloss = agent.optimize_q_network(args.batch_size, args.gamma)
                    if newloss is not None:
                        loss += newloss
                state = next_state
                cand = next_cand
                t = count
                count+=1
                if done:
                    # every episode update the target model to be same with model
                    if reward.item() == 1:  # recommend successfully
                        if t < 5:
                            SR5 += 1
                            SR10 += 1
                            SR15 += 1
                        elif t < 10:
                            SR10 += 1
                            SR15 += 1
                        else:
                            SR15 += 1
                        Rank += (1/math.log(t+3,2) + (1/math.log(t+2,2)-1/math.log(t+3,2))/math.log(done+1,2))
                    else:
                        Rank += 0
                    AvgT += t+1
                    total_reward += epi_reward
                    buffer[i_episode%args.num_rollouts][1]=t+1
                    buffer[i_episode%args.num_rollouts][2]=epi_reward
                    break
            #min_turn = 15
            #min_x = 20
            #for x, temps in buffer.items():
            #    num_turns = temps[1]
            #    if num_turns<=min_turn:
            #        min_turn=num_turns
            #        min_x = x
            #good_trajectory = buffer[min_x][0]
            #for temp in good_trajectory:
            #    agent.memory.push(temp[0], temp[1], temp[2], temp[3], temp[4])
            # back-propagation
            R = current_node.r
            current_node.update(R)
            while current_node.parent is not None: # loop back-propagation until root is reached
                current_node = current_node.parent
                R = current_node.r + args.gamma * R
                current_node.update(R)
            if (i_episode+1)%args.num_rollouts==0:
                total_buffer.add(buffer)
                newloss = agent.optimize_policy_network(args.batch_size, total_buffer)
                if newloss is not None:
                    loss += newloss
        enablePrint() # Enable print function
        print('loss : {} in epoch_user {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, rewards:{} '
                  'Total epoch_user:{}'.format(SR5 / args.sample_times, SR10 / args.sample_times, SR15 / args.sample_times,
                                                AvgT / args.sample_times, Rank / args.sample_times, total_reward / args.sample_times, args.sample_times))

        if train_step % args.eval_num == 0:
            SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean = dqn_evaluate(args, kg, dataset, agent, filename, train_step)
            if SR15_mean>SR15_best:
                SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best=SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean
            print("best!!!!!!!!!SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}!!!!".format(
                SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best))
            print('current epoch:******',train_step,'******')
            test_performance.append(SR15_mean)
        if train_step % args.save_num == 0:
            agent.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
    print(test_performance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--w', type=float, default=1.5, help='balance between exploration and exploitation.')
    parser.add_argument('--num_rollouts', type=int, default=20, help='number of rollouts for each user')
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=10000, help='size of memory ')
    parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM_STAR, YELP_STAR, BOOK, MOVIE],
                        help='One of {LAST_FM_STAR, YELP_STAR, BOOK, MOVIE}.')
    parser.add_argument('--entropy_method', type=str, default='entropy', choices=['entropy', 'weight entropy','match'], help='entropy_method is one of {entropy, weight entropy,match}')
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')
    parser.add_argument('--sample_times', type=int, default=300, help='the epoch of sampling')
    parser.add_argument('--observe_num', type=int, default=100, help='the observe_num')
    parser.add_argument('--max_steps', type=int, default=100, help='max training steps')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
    parser.add_argument('--fea_score', type=str, default='entropy', choices=['entropy','match'], help='feature score')
    parser.add_argument('--fix_emb', action='store_false', help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='transe', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    for user in kg.G['user']:
        acc = []
        for item in kg.G['user'][user]['interact']:
            for fea in kg.G['item'][item]['belong_to']:
                acc.append(fea)
        kg.G['user'][user]['acc'] = set(acc)
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name)
    filename = 'train-data-{}'.format(args.data_name)
    train(args, kg, dataset, filename)

if __name__ == '__main__':
    main()

