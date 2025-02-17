
import json
import numpy as np
import os
import random
from utils import *
from torch import nn
import ipdb
from tkinter import _flatten
from collections import Counter
import copy


class MultiChoiceRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, embed, ask_large_fea_num=4, classify=True, seed=1, max_turn=15, cand_num=10, cand_item_num=10, attr_num=20, mode='train', ask_num=1, entropy_way='weight entropy', fm_epoch=0, fea_score="entropy"):
        self.data_name = data_name
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn
        self.attr_state_num = attr_num
        self.kg = kg
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len
        self.large_feature_length=42
        self.other_feature=self.large_feature_length+1
        self.small_feature_to_large={}
        self.get_feature_dict()
        self.fea_score=fea_score
        self.ask_num = ask_num
        self.rec_num = 10
        self.random_sample_feature = False
        self.random_sample_item = False
        if cand_num == 0:
            self.cand_num = 10
            self.random_sample_feature = True
        else:
            self.cand_num = cand_num
        if cand_item_num == 0:
            self.cand_item_num = 10
            self.random_sample_item = True
        else:
            self.cand_item_num = cand_item_num
        self.ent_way = entropy_way
        self.reachable_feature = []
        self.user_acc_feature = []
        self.user_rej_feature = []
        self.cand_items = []
        self.rej_item=[]
        self.item_feature_pair = {}
        self.cand_item_score = []
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0
        self.cur_node_set = []
        self.user_embed = None
        self.conver_his = []
        self.attr_ent = []
        self.ui_dict,self.u_multi = self.__load_rl_data__(data_name, mode=mode)
        self.user_weight_dict = dict()
        self.user_items_dict = dict()
        set_random_seed(self.seed)
        if mode == 'train':
            self.__user_dict_init__()
        elif mode == 'test':
            self.ui_array = None
            self.__test_tuple_generate__()
            self.test_num = 0
        embeds = load_embed(data_name, embed, epoch=fm_epoch)
        if len(embeds)!=0:
            self.ui_embeds=embeds['ui_emb']
            self.feature_emb = embeds['feature_emb']
        else:
            self.ui_embeds = nn.Embedding(self.user_length+self.item_length, 64).weight.data.numpy()
            self.feature_emb = nn.Embedding(self.feature_length, 64).weight.data.numpy()
        self.action_space = 2

        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'ask_large_suc': 0.02,
            'ask_large_fail': -0.2,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'ask_large_suc': 1.5,
            'ask_large_fail': -1.5,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()
        self.classify = classify
        self.acc_large_fea = [] 
        self.rej_large_fea = [] 
        self.asked_large_fea = [] 
        self.ask_large_fea_num = ask_large_fea_num 

    def __load_rl_data__(self, data_name, mode):
        if mode == 'train':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
            with open(os.path.join(DATA_DIR[data_name], 'UI_data/train.pkl'), 'rb') as f:
                u_multi=pickle.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'), encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
            with open(os.path.join(DATA_DIR[data_name], 'UI_data/test.pkl'), 'rb') as f:
                u_multi=pickle.load(f)
        return mydict,u_multi


    def __user_dict_init__(self):
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums
        print('user_dict init successfully!')

    def get_feature_dict(self):
        num=0
        for m in self.kg.G['feature']:
            if len(self.kg.G['feature'][m]['link_to_feature']):
                large=self.kg.G['feature'][m]['link_to_feature'][0]
                self.small_feature_to_large[m]=large
            else:
                self.small_feature_to_large[m]=self.other_feature
                num+=1
            

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.u_multi.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        self.ui_array = np.array(ui_list, dtype=object)
        np.random.shuffle(self.ui_array)
    
    def get_sameatt_items(self):
        users = list(self.ui_dict.keys())
        self.ui_satt_items={}

        for user in users:
            user=int(user)
            all_items = self.ui_dict[str(user)]
            same_att_items={}
            a2i,i2a = {},{}
            for item in all_items:
                att=set(self.kg.G['item'][item]['belong_to'])
                i2a[item]=att
                for a in att:
                    if a in a2i:
                        a2i[a].append(item)
                    else:
                        a2i[a]=[item]

            for item in all_items:
                can_att=i2a[item]
                can_items=[]
                for a in can_att:
                    tmp_items=a2i[a]
                    can_items+=tmp_items
                same_att_items[item]=can_items

            self.ui_satt_items[user]=same_att_items
            
    def reset_by_user_id_and_state(self, user_id, acc_feature, current_target_item, embed=None):
        self.acc_large_fea = [] 
        self.rej_large_fea = [] 
        self.asked_large_fea = [] 
        self.large_feature_groundtrue = []     
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]
        self.cur_conver_step = 0
        self.cur_node_set = []
        self.rej_item=[]
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            self.user_id = user_id
            self.target_item = copy.deepcopy(current_target_item)
        elif self.mode == 'test':
            self.user_id = user_id
            self.target_item  = self.ui_array[self.test_num, 1]
            self.test_num += 1
        print('-----------reset state vector------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        feature_groundtrue=[]
        for i in self.target_item:
            feature_groundtrue += self.kg.G['item'][i]['belong_to']
        self.feature_groundtrue=list(set(feature_groundtrue))
        self.reachable_feature = []  
        self.user_acc_feature = []  
        self.user_special_acc_feature = []
        self.user_rej_feature = []
        self.user_special_rej_feature = []
        self.cand_items = list(range(self.item_length))
        self.user_embed = self.ui_embeds[self.user_id].tolist()
        self.conver_his = [0] * self.max_turn
        self.attr_ent = [0] * self.attr_state_num
        attrs=set(self.kg.G['item'][self.target_item[0]]['belong_to'])
        for i in range(1,len(self.target_item)):
            attrs2=set(self.kg.G['item'][self.target_item[i]]['belong_to'])
            attrs=attrs&attrs2
        attrs=list(attrs)
        user_like_random_fea = copy.deepcopy(acc_feature[0])
        self.user_acc_feature.append(user_like_random_fea)
        if self.classify:
            if user_like_random_fea not in self.kg.G['user'][self.user_id]['acc']:
                self.user_special_acc_feature.append(user_like_random_fea)
        self.cur_node_set.append(user_like_random_fea)
        self._update_cand_items([user_like_random_fea], acc_rej=True)
        self._updata_reachable_feature()
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1
        print('=== init user prefer feature: {}'.format(self.cur_node_set))
        self._update_feature_entropy()
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.cand_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            if max_ind in max_ind_list:
                break
            max_ind_list.append(max_ind)
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.remove(v) for v in max_fea_id]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        for fea in self.feature_groundtrue:
            if fea != user_like_random_fea:
                self.large_feature_groundtrue.append(self.small_feature_to_large[fea])
        self.large_feature_groundtrue = list(set(self.large_feature_groundtrue))
        return self._get_state(), self._get_cand(), self._get_action_space()

    def reset_by_user_id(self, user_id, embed=None):
        self.acc_large_fea = [] 
        self.rej_large_fea = [] 
        self.asked_large_fea = [] 
        self.large_feature_groundtrue = []
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]
        self.cur_conver_step = 0
        self.cur_node_set = []
        self.rej_item=[]
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            self.user_id = user_id
            self.target_item = random.choice(self.u_multi[str(self.user_id)])
        elif self.mode == 'test':
            self.user_id = user_id
            self.target_item  = self.ui_array[self.test_num, 1]
            self.test_num += 1
        print('-----------reset state vector------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        feature_groundtrue=[]
        for i in self.target_item:
            feature_groundtrue += self.kg.G['item'][i]['belong_to']
        self.feature_groundtrue=list(set(feature_groundtrue))
        self.reachable_feature = []
        self.user_acc_feature = []
        self.user_special_acc_feature = []
        self.user_rej_feature = []
        self.user_special_rej_feature = []
        self.cand_items = list(range(self.item_length))
        self.user_embed = self.ui_embeds[self.user_id].tolist()
        self.conver_his = [0] * self.max_turn
        self.attr_ent = [0] * self.attr_state_num
        attrs=set(self.kg.G['item'][self.target_item[0]]['belong_to'])
        for i in range(1,len(self.target_item)):
            attrs2=set(self.kg.G['item'][self.target_item[i]]['belong_to'])
            attrs=attrs&attrs2
        attrs=list(attrs)
        user_like_random_fea = random.choice(attrs)
        self.user_acc_feature.append(user_like_random_fea)
        if self.classify:
            if user_like_random_fea not in self.kg.G['user'][self.user_id]['acc']:
                self.user_special_acc_feature.append(user_like_random_fea)
        self.cur_node_set.append(user_like_random_fea)
        self._update_cand_items([user_like_random_fea], acc_rej=True)
        self._updata_reachable_feature()
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1
        print('=== init user prefer feature: {}'.format(self.cur_node_set))
        self._update_feature_entropy()
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.cand_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            if max_ind in max_ind_list:
                break
            max_ind_list.append(max_ind)
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.remove(v) for v in max_fea_id]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        for fea in self.feature_groundtrue:
            if fea != user_like_random_fea:
                self.large_feature_groundtrue.append(self.small_feature_to_large[fea])
        self.large_feature_groundtrue = list(set(self.large_feature_groundtrue))
        return self._get_state(), self._get_cand(), self._get_action_space()

        
    def reset(self, embed=None):
        self.acc_large_fea = [] 
        self.rej_large_fea = [] 
        self.asked_large_fea = [] 
        self.large_feature_groundtrue = []
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]
        self.cur_conver_step = 0
        self.cur_node_set = []
        self.rej_item=[]
        if self.mode == 'train':
            users = list(self.user_weight_dict.keys())
            self.user_id = np.random.choice(users)
            self.target_item = random.choice(self.u_multi[str(self.user_id)])
        elif self.mode == 'test':
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item  = self.ui_array[self.test_num, 1]
            self.test_num += 1
        print('-----------reset state vector------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        feature_groundtrue=[]
        for i in self.target_item:
            feature_groundtrue += self.kg.G['item'][i]['belong_to']
        self.feature_groundtrue=list(set(feature_groundtrue))
        self.reachable_feature = []
        self.user_acc_feature = []
        self.user_special_acc_feature = []
        self.user_rej_feature = []
        self.user_special_rej_feature = []
        self.cand_items = list(range(self.item_length))
        self.user_embed = self.ui_embeds[self.user_id].tolist()
        self.conver_his = [0] * self.max_turn
        self.attr_ent = [0] * self.attr_state_num
        attrs=set(self.kg.G['item'][self.target_item[0]]['belong_to'])
        for i in range(1,len(self.target_item)):
            attrs2=set(self.kg.G['item'][self.target_item[i]]['belong_to'])
            attrs=attrs&attrs2
        attrs=list(attrs)
        user_like_random_fea = random.choice(attrs)
        self.user_acc_feature.append(user_like_random_fea)
        if self.classify:
            if user_like_random_fea not in self.kg.G['user'][self.user_id]['acc']:
                self.user_special_acc_feature.append(user_like_random_fea)
        self.cur_node_set.append(user_like_random_fea)
        self._update_cand_items([user_like_random_fea], acc_rej=True)
        self._updata_reachable_feature()
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1
        print('=== init user prefer feature: {}'.format(self.cur_node_set))
        self._update_feature_entropy()
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.cand_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            if max_ind in max_ind_list:
                break
            max_ind_list.append(max_ind)
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.remove(v) for v in max_fea_id]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        for fea in self.feature_groundtrue:
            if fea != user_like_random_fea:
                self.large_feature_groundtrue.append(self.small_feature_to_large[fea])
        self.large_feature_groundtrue = list(set(self.large_feature_groundtrue))
        return self._get_state(), self._get_cand(), self._get_action_space()

    def _get_cand(self):
        if self.random_sample_feature:
            cand_feature = self._map_to_all_id(random.sample(self.reachable_feature, min(len(self.reachable_feature),self.cand_num)),'feature')
        else:
            cand_feature = self._map_to_all_id(self.reachable_feature[:self.cand_num],'feature')
        if self.random_sample_item:
            cand_item =  self._map_to_all_id(random.sample(self.cand_items, min(len(self.cand_items),self.cand_item_num)),'item')
        else:
            cand_item = self._map_to_all_id(self.cand_items[:self.cand_item_num],'item')
        #cand = cand_feature + cand_item
        #return cand
        return (cand_feature,  cand_item)
    
    def _get_action_space(self):
        action_space = [self._map_to_all_id(self.reachable_feature,'feature'), self._map_to_all_id(self.cand_items,'item')]
        return action_space

    def _get_state(self):
        if self.data_name in ['YELP_STAR']:
            self_cand_items = self.cand_items[:5000]
            set_cand_items = set(self_cand_items)
        else:
            self_cand_items = self.cand_items        
        user = [self.user_id]
        cur_node = [x + self.user_length + self.item_length for x in self.cur_node_set] 
        cand_items = [x + self.user_length for x in self_cand_items]
        reachable_feature = [x + self.user_length + self.item_length for x in self.reachable_feature]
        neighbors = cur_node + user + cand_items + reachable_feature
        idx = dict(enumerate(neighbors))
        idx = {v: k for k, v in idx.items()}
        i = []
        v = []
        for item in self_cand_items:
            item2 = item + self.user_length
            for fea in self.item_feature_pair[item]:
                fea2 = fea + self.user_length + self.item_length
                i.append([idx[item2], idx[fea2]])
                i.append([idx[fea2], idx[item2]])
                v.append(1)
                v.append(1)

        user_idx = len(cur_node)
        cand_item_score = self.sigmoid(self.cand_item_score)
        for item, score in zip(self.cand_items, cand_item_score):
            if self.data_name in ['YELP_STAR']:
                if item not in set_cand_items:
                    continue
            item += self.user_length
            i.append([user_idx, idx[item]])
            i.append([idx[item], user_idx])
            v.append(score)
            v.append(score)
        
        i = torch.LongTensor(i)
        v = torch.FloatTensor(v)
        neighbors = torch.LongTensor(neighbors)
        adj = torch.sparse.FloatTensor(i.t(), v, torch.Size([len(neighbors),len(neighbors)]))

        state = {'cur_node': cur_node,
                 'neighbors': neighbors,
                 'adj': adj,
                 'rej_feature': [x + self.user_length + self.item_length for x in self.user_rej_feature],
                 'rej_item': self.rej_item,
                 'user':self.user_id,
                 'special_acc_feature':[x + self.user_length + self.item_length for x in self.user_special_acc_feature],
                 'special_rej_feature':[x + self.user_length + self.item_length for x in self.user_special_rej_feature]
                 }
        return state
    
    def step(self, action, sorted_actions, embed=None):  
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]

        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif len(self.acc_large_fea) == 0 and self.data_name!='MOVIE':
            reward = 0
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_large_fail']
            score={}
            large_small={}
            for i in range(len(sorted_actions)):
                act=sorted_actions[i]
                if act < self.user_length + self.item_length:
                    continue
                small_fea=self._map_to_old_id(act)
                large=self.small_feature_to_large[small_fea]
                if large in score:
                    score[large]+=1/(i+1)
                    large_small[large].append(small_fea)
                else:
                    score[large]=0.0
                    large_small[large]=[]
                    score[large]+=1/(i+1)
                    large_small[large].append(small_fea)
            for large_score in sorted(score.items(),key=lambda x :-x[1])[:self.ask_large_fea_num]:
                large = large_score[0]
                if large not in self.asked_large_fea:
                    self.asked_large_fea.append(large)
                    if large in self.large_feature_groundtrue:
                        reward += self.reward_dict['ask_large_suc']
                        self.conver_his[self.cur_conver_step] = self.history_dict['ask_large_suc']
                        self.acc_large_fea.append(large)
                    else:
                        reward += self.reward_dict['ask_large_fail']
                        self.rej_large_fea.append(large)
            if self.classify:
                for fea in self.kg.G['user'][self.user_id]['acc']:
                    if self.small_feature_to_large[fea] in self.rej_large_fea:
                        self.user_special_rej_feature.append(fea)
            cand_items = []
            for item in self.cand_items:
                smalls = self.kg.G['item'][item]['belong_to']
                larges = set([self.small_feature_to_large[x] for x in smalls])
                if larges & set(self.acc_large_fea) != set():
                    cand_items.append(item)            
            if len(cand_items)!=0:
                self.cand_items=cand_items
            cand_item_score = self._item_score()
            item_score_tuple = list(zip(self.cand_items, cand_item_score))
            sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
            self.cand_items, self.cand_item_score = zip(*sort_tuple)
        elif action >= self.user_length + self.item_length:
            asked_feature = []
            for i in range(len(sorted_actions)):
                act=sorted_actions[i]
                if act < self.user_length + self.item_length:
                    continue
                else:
                    asked_feature.append(self._map_to_old_id(act))
            asked_feature = asked_feature[:self.cand_num]
            reward, done, acc_rej = self._ask_update(asked_feature)
            self._update_cand_items(asked_feature, acc_rej)
        else:
            recom_items = []
            recom_items_gnn_id=[]
            for act in sorted_actions:
                if act < self.user_length + self.item_length:
                    recom_items_gnn_id.append(act)
                    recom_items.append(self._map_to_old_id(act))
                    if len(recom_items) == self.rec_num:
                        break
            reward, done = self._recommend_update(recom_items)
            if reward > 0:
                print('-->Recommend successfully!')
            else:
                self.rej_item+=recom_items_gnn_id
                print('-->Recommend fail !')
        
        self._updata_reachable_feature()

        print('reachable_feature num: {}'.format(len(self.reachable_feature)))
        print('cand_item num: {}'.format(len(self.cand_items)))

        self._update_feature_entropy()
        if len(self.reachable_feature) != 0:
            reach_fea_score = self._feature_score()

            max_ind_list = []
            for k in range(self.cand_num):
                max_score = max(reach_fea_score)
                max_ind = reach_fea_score.index(max_score)
                reach_fea_score[max_ind] = 0
                if max_ind in max_ind_list:
                    break
                max_ind_list.append(max_ind)
            max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
            [self.reachable_feature.remove(v) for v in max_fea_id]
            [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        self.cur_conver_step += 1
        return self._get_state(), self._get_cand(), self._get_action_space(), reward, done


    def _updata_reachable_feature(self):
        next_reachable_feature = []
        reachable_item_feature_pair = {}
        for cand in self.cand_items:
            fea_belong_items = list(self.kg.G['item'][cand]['belong_to'])
            fea_belong_items2 = []
            if self.acc_large_fea != []:
                for fea in fea_belong_items:
                    if self.small_feature_to_large[fea] in self.acc_large_fea:
                        fea_belong_items2.append(fea)
                fea_belong_items = fea_belong_items2
            elif self.rej_large_fea != []:
                for fea in fea_belong_items:
                    if self.small_feature_to_large[fea] not in self.rej_large_fea:
                        fea_belong_items2.append(fea)
                fea_belong_items = fea_belong_items2
            next_reachable_feature.extend(fea_belong_items)
            reachable_item_feature_pair[cand] = list(set(fea_belong_items) - set(self.user_rej_feature))
            next_reachable_feature = list(set(next_reachable_feature))
        self.reachable_feature = list(set(next_reachable_feature) - set(self.user_acc_feature) - set(self.user_rej_feature))
        self.item_feature_pair = reachable_item_feature_pair

    def _feature_score(self):
        
        reach_fea_score = []
        if self.fea_score=="entropy":
            for feature_id in self.reachable_feature:

                score = self.attr_ent[feature_id]
                reach_fea_score.append(score)


        else:
            for feature_id in self.reachable_feature:
                fea_embed = self.feature_emb[feature_id]
                score = 0
                score += np.inner(np.array(self.user_embed), fea_embed)
                prefer_embed = self.feature_emb[self.user_acc_feature, :]
                rej_embed = self.feature_emb[self.user_rej_feature, :]
                for i in range(len(self.user_acc_feature)):
                    score += np.inner(prefer_embed[i], fea_embed)
                for i in range(len(self.user_rej_feature)):
                    score -= self.sigmoid([np.inner(rej_embed[i], fea_embed)])
                reach_fea_score.append(score)
            
        return reach_fea_score


    def _item_score(self):
        cand_item_score = []
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]
            score = 0
            score += np.inner(np.array(self.user_embed), item_embed)
            prefer_embed = self.feature_emb[self.user_acc_feature, :]
            unprefer_feature = list(set(self.user_rej_feature) & set(self.kg.G['item'][item_id]['belong_to']))
            for fea in self.kg.G['item'][item_id]['belong_to']:
                if self.small_feature_to_large[fea] in self.rej_large_fea:
                    unprefer_feature.append(fea)
            unprefer_feature = list(set(unprefer_feature))
            unprefer_embed = self.feature_emb[unprefer_feature, :]
            for i in range(len(self.user_acc_feature)):
                score += np.inner(prefer_embed[i], item_embed)
            if False:
                special_prefer_embed = self.feature_emb[self.user_special_acc_feature, :]
                for i in range(len(self.user_special_acc_feature)):
                    score += np.inner(special_prefer_embed[i], item_embed)
            for i in range(len(unprefer_feature)):
                score -= self.sigmoid([np.inner(unprefer_embed[i], item_embed)])[0]
                if self.classify:
                    if i in range(len(self.user_special_rej_feature)):
                        score -= self.sigmoid([np.inner(unprefer_embed[i], item_embed)])[0]
            cand_item_score.append(score)
        return cand_item_score


    def _ask_update(self, asked_features):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0
        reward=0
        acc_rej = False
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']
        for asked_feature in asked_features:
            if asked_feature in self.feature_groundtrue:
                acc_rej = True
                self.user_acc_feature.append(asked_feature)
                self.cur_node_set.append(asked_feature)
                reward += self.reward_dict['ask_suc']
                self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
                if self.classify:
                    if asked_feature not in self.kg.G['user'][self.user_id]['acc']:
                            self.user_special_acc_feature.append(asked_feature)
            else:
                if self.classify:
                    if asked_feature in self.kg.G['user'][self.user_id]['acc']:
                            self.user_special_rej_feature.append(asked_feature)
                self.user_rej_feature.append(asked_feature)
                reward += self.reward_dict['ask_fail']
        if self.cand_items == []:
            done = 1
            reward = self.reward_dict['cand_none']

        return reward, done, acc_rej

    def _update_cand_items(self, asked_feature, acc_rej):
        acc_item=[]
        rej_item=[]
        for fea in asked_feature:
            if fea in self.feature_groundtrue:
                print('=== ask acc {}: update cand_items'.format(fea))
                feature_items = self.kg.G['feature'][fea]['belong_to']
                cand_items = set(self.cand_items) & set(feature_items)
                acc_item+= list(cand_items)
            else:
                pass
        if len(acc_item)==0:
            cand_items=list(set(self.cand_items)-(set(self.cand_items) & set(rej_item)))
        else:
            cand_items=list(set(acc_item)-(set(acc_item) & set(rej_item)))
        if len(cand_items)!=0:
            self.cand_items=cand_items
        cand_item_score = self._item_score()
        item_score_tuple = list(zip(self.cand_items, cand_item_score))
        sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
        self.cand_items, self.cand_item_score = zip(*sort_tuple)
    
    def _recommend_update(self, recom_items):
        print('-->action: recommend items')
        print(set(recom_items) - set(self.cand_items[: self.rec_num]))
        self.cand_items = list(self.cand_items)
        self.cand_item_score = list(self.cand_item_score)
        hit=False
        for i in self.target_item:
            if i in recom_items:
                hit=True
                break
        if hit:
            reward = self.reward_dict['rec_suc'] 
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu']
            tmp_score = []
            for item in recom_items:
                idx = self.cand_items.index(item)
                tmp_score.append(self.cand_item_score[idx])
            self.cand_items = recom_items
            self.cand_item_score = tmp_score
            done = recom_items.index(i) + 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']
            if len(self.cand_items) > self.rec_num:
                for item in recom_items:
                    del self.item_feature_pair[item]
                    idx = self.cand_items.index(item)
                    self.cand_items.pop(idx)
                    self.cand_item_score.pop(idx)
            done = 0
        return reward, done

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent
        elif self.ent_way == 'weight_entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score_sig = self.sigmoid(self.cand_item_score)
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            sum_score_sig = sum(cand_item_score_sig)

            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                p2 = 1.0 - p1
                if p1 == 1 or p1 <= 0:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def _map_to_all_id(self, x_list, old_type):
        if old_type == 'item':
            return [x + self.user_length for x in x_list]
        elif old_type == 'feature':
            return [x + self.user_length + self.item_length for x in x_list]
        else:
            return x_list

    def _map_to_old_id(self, x):
        if x >= self.user_length + self.item_length:
            x -= (self.user_length + self.item_length)
        elif x >= self.user_length:
            x -= self.user_length
        return x

