# -*- coding: utf-8 -*-
# import the necessary packages
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from freq import *
import time
from mlp_dropout import MLPClassifier, MLPRegression
from DGCNN_embedding import DGCNN
import os
np.random.seed(0)
torch.manual_seed(0)
# 1. Define some Hyper Parameters
BATCH_SIZE = 5    # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9      # epsilon used for epsilon greedy approach
GAMMA = 0.6         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 50       # How frequently target netowrk updates
MEMORY_CAPACITY = 200                 # The capacity of experience replay buffer

N_ACTIONS = 6
N_STATES = 6 # 6 states
ENV_A_SHAPE = 0     # to confirm the shape


class Data(object):
    def __init__(self,state,state_next,reward,action):
        self.state_now = state
        self.state_next = state_next
        self.reward = reward
        self.action = action

# 2. Define the network used in both target net and the net for training
class Net(nn.Module):
    def __init__(self,mama_lidu,action_number):
        # Define the network structure, a very simple fully connected network
        super(Net, self).__init__()
        model = DGCNN
        self.edge_feat_dim = 0

        if mama_lidu == 'family':
            self.num_node_feats = 11
            self.action_num = action_number
        elif mama_lidu == 'package':
            self.num_node_feats = 386
            self.action_num = action_number
        elif mama_lidu == 'class':
            self.num_node_feats = 2431
            self.action_num = action_number
        self.gnn = model(latent_dim=[32, 32, 32, 1],
                            output_dim=0, 
                            num_node_feats= self.num_node_feats,
                            num_edge_feats=0,
                            k=6, 
                            )
 
        out_dim = self.gnn.dense_dim
        # print()
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=128, num_class=self.action_num, with_dropout=True)
        self.regression = False

        self.mode = 'cpu'
        # # Define the structure of fully connected network
        # self.fc1 = nn.Linear(N_STATES, 128)  # layer 1
        # self.fc1.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc1
        # self.fc2 = nn.Linear(128, 64)  # layer 1
        # self.fc2.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc1
        # # self.fc3 = nn.Linear(256, 128)  # layer 1
        # # self.fc3.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc1
        # self.out = nn.Linear(64, N_ACTIONS) # layer 2
        # self.out.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc2
    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if self.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, self.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if self.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels
        
        
    def forward(self, batch_graph):
        # print(batch_graph)
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label

        embed = self.gnn(batch_graph, node_feat, edge_feat)

        return self.mlp(embed, labels)


        # return actions_value
        
# 3. Define the DQN network and its corresponding methods
class DQN(object):
    def __init__(self,mama_lidu,action_number):
        if mama_lidu == 'family':
            self.num_node_feats = 11
            self.action_num = action_number
        elif mama_lidu == 'package':
            self.num_node_feats = 386
            self.action_num = action_number
        elif mama_lidu == 'class':
            self.num_node_feats = 2431
        # -----------Define 2 networks (target and training)------#
        self.eval_net, self.target_net = Net(mama_lidu,self.action_num), Net(mama_lidu,self.action_num)
        # Define counter, memory size and loss function
        self.learn_step_counter = 0 # count the steps of learning process
        self.memory_counter = 0 # counter used for experience replay buffer
        
        # ----Define the memory (or the buffer), allocate some space to it. The number 
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = []
        
        #------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        
        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()
        
    def  choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy
        action = np.zeros(self.action_num)
        # x = torch.unsqueeze(torch.FloatTensor(x), 0) # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            # use epsilon-greedy approach to take action
            # actions_value = self.eval_net.forward(x).data.numpy()[0]
            # print(self.eval_net(x))
            # input()
            index = np.argmax(self.eval_net(x)[0].data.numpy())
             
            # actions_value = self.action_dict[index]
            # actions_value = actions_value
            #print(actions_value)

            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            # action[1] = int(np.argmax(actions_value[2:-1]))
            action[int(index)] = 1
            # action[int(np.argmax(actions_value[2:])) +2] = 1

            # print(actions_value,action) 

            #print(action)
        else:   # random
            index = random.randint(0,self.action_num-1)
            action[index] = 1
            # action[random.randint(0,3)+2] = 1
            
        return action
    
    def  random_choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy
        action = np.zeros(self.action_num)
        index = random.randint(0,self.action_num-1)
        action[index] = 1
            # action[random.randint(0,3)+2] = 1
            
        return action        
    def store_transition(self, data):
        # This function acts as experience replay buffer        
        # transition = np.hstack((s, a, [r], s_)) # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory.append(data)
        self.memory_counter += 1
        
    
        
    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.
        
        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) # randomly select some data from buffer
        # extract experiences of batch size from buffer.

        b_memory =  [self.memory[i] for i in sample_index]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = [i.state_now for i in b_memory]
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor([i.action for i in b_memory]))
        b_r = Variable(torch.FloatTensor([i.reward for i in b_memory]))
        b_s_ = [i.state_next for i in b_memory]
        
        # calculate the Q value of state-action pair

        q_eval= self.eval_net(b_s)[0] # (batch_size, 1)
        # print(q_eval.shape)
        # calculate the q value of next state
        q_next = self.target_net(b_s_)[0].detach() # detach from computational graph, don't back propagate

        q_target = torch.rand((q_eval.shape[0],q_eval.shape[1]))
        q_target.copy_(q_eval)


        for i in range(BATCH_SIZE):
            q_target[i,[torch.argmax(b_a[i, :])]] = b_r[i] + 0.1 * (q_next[i, :].max())

 
        # print(q_target)

        loss = self.loss_func(q_eval, q_target)
        #print(loss)
        self.optimizer.zero_grad() # reset the gradient to zero
        loss.backward()
        self.optimizer.step() # execute back propagation for one step
        q_eval = self.eval_net(b_s)[0].gather(1, b_a) 
        # print(q_eval)    
        # input() 
'''
--------------Procedures of DQN Algorithm------------------
'''
# create the object of DQN class
def test(classifier,data,clf_type, mama_lidu,file,attack_method):
    # print(dataset)
    # input()

        file_name_y = os.path.join(file,'y_pred_record_'+str(attack_method)+'.txt')
        file_name_action = os.path.join(file,'action_record_'+str(attack_method)+'.txt')
        file_name_infor = os.path.join(file,'attack_infomation_'+str(attack_method)+'.txt')
        if  os.path.exists(file_name_y):
            os.remove(file_name_y)
        if  os.path.exists(file_name_action):
            os.remove(file_name_action)
        if  os.path.exists(file_name_infor):
            os.remove(file_name_infor)
        begin  = time.time()
        env = freq(mama_lidu) # Use cartpole game as environment
        # print(env.action_num)
        # input()
        dqn = DQN(mama_lidu,env.action_num)
        print("\nCollecting experience...")
        ep_r = 0
        is_end = 1
        env.reset(data,mama_lidu,classifier,clf_type)
        data_tem = np.copy(data)
        # input()

        if attack_method == 'SRI_no':
            for i_episode in range(400):
                if is_end == 1:
                    env.reset(data,mama_lidu,classifier,clf_type)
                    data_tem = np.copy(data)
                    is_end = 0
                    s = env.state

                    a_nops = dqn.random_choose_action([s])

                else:
                    s = s_
                    a_nops = dqn.random_choose_action([s])  


                data_tem,s_,is_end,r,y_pred,per_num = env.getReward(a_nops,data_tem,mama_lidu,classifier,attack_method,clf_type)
                # if '0069B' in file:
                # print(s_.call_times)
                # print(y_pred)
                # input()
                # break
                with open(file_name_y,'a',encoding='utf-8') as f:
                    text =  str(y_pred)
                    f.write(text)
                    f.write('\r\n')
                    # print(file_name)
                    # input()

                with open(file_name_action,'a',encoding='utf-8') as f:
                    text =  str(np.argmax(a_nops))
                    f.write(text)
                    f.write('\r\n')
                if is_end == 1:
                    end = time.time()

                    with open(file_name_infor,'a',encoding='utf-8') as f:
                        f.write(str(per_num))
                        f.write('   ')
                        f.write(str(end - begin))
                        f.write('\r\n')
                    print('attack success and the pertubation is:',per_num)
                    break
            if is_end == 0:
                end = time.time()
                with open(file_name_infor,'a',encoding='utf-8') as f:
                    f.write(str(per_num))
                    f.write('   ')
                    f.write(str(y_pred))
                    f.write('   ')
                    f.write(str(end - begin))
                    f.write('\r\n')
                print('attack fail and the pertubation is:',per_num)
        elif attack_method == 'SRI':
            for i_episode in range(400):
                if is_end == 1:
                    env.reset(data,mama_lidu,classifier,clf_type)
                    data_tem = np.copy(data)
                    is_end = 0
                    s = env.state

                    a_nops = dqn.random_choose_action([s])

                else:
                    s = s_
                    a_nops = dqn.random_choose_action([s])  


                data_tem,s_,is_end,r,y_pred,per_num = env.getReward(a_nops,data_tem,mama_lidu,classifier,attack_method,clf_type)
                if r > 0.1:
                    s = s_
                    with open(file_name_y,'a',encoding='utf-8') as f:
                        text =  str(y_pred)
                        f.write(text)
                        f.write('\r\n')
                        # print(file_name)
                        # input()

                    with open(file_name_action,'a',encoding='utf-8') as f:
                        text =  str(np.argmax(a_nops))
                        f.write(text)
                        f.write('\r\n')
                else:
                    s_ = s


                if is_end == 1:
                    end = time.time()

                    with open(file_name_infor,'a',encoding='utf-8') as f:
                        f.write(str(per_num))
                        f.write('   ')
                        f.write(str(y_pred))
                        f.write('   ')
                        f.write(str(end - begin))
                        f.write('\r\n')
                    print('attack success and the pertubation is:',per_num)
                    break
            if is_end == 0:
                end = time.time()
                with open(file_name_infor,'a',encoding='utf-8') as f:
                    per_num = 500
                    f.write(str(per_num))
                    f.write('   ')
                    f.write(str(y_pred))
                    f.write('   ')
                    f.write(str(end - begin))
                    f.write('\r\n')
                print('attack fail and the pertubation is:',per_num)

        else:
            for i_episode in range(2500):
                # if i_episode
                # print(i_episode)
                if is_end == 1:
                    env.reset(data,mama_lidu,classifier,clf_type)
                    data_tem = np.copy(data)
                    is_end = 0
                    s = env.state

                    a_nops = dqn.choose_action([s])

                else:
                    s = s_
                    a_nops = dqn.choose_action([s])  


                data_tem,s_,is_end,r,y_pred,per_num = env.getReward(a_nops,data_tem,mama_lidu,classifier,attack_method,clf_type)
                # print(r)
                data_store = Data(s,s_,r, a_nops)
                # store the transitions of states
                dqn.store_transition(data_store)

                # ep_r += r_1

                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()

                if(i_episode % 100 == 0):
                    # print(y_pred)
                    # print(a_nops)
                    # print(r)
                    print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
            is_end = 1 
            # input()      
            for i_episode in range(400):
                if is_end == 1:
                    env.reset(data,mama_lidu,classifier,clf_type)
                    data_tem = np.copy(data)
                    is_end = 0
                    s = env.state

                    a_nops = dqn.choose_action([s])

                else:
                    s = s_
                    a_nops = dqn.choose_action([s])  


                data_tem,s_,is_end,r,y_pred,per_num = env.getReward(a_nops,data_tem,mama_lidu,classifier,attack_method,clf_type)



                with open(file_name_y,'a',encoding='utf-8') as f:
                    text =  str(y_pred)
                    f.write(text)
                    f.write('\r\n')
                    # print(file_name)
                    # input()

                with open(file_name_action,'a',encoding='utf-8') as f:
                    text =  str(np.argmax(a_nops))
                    f.write(text)
                    f.write('\r\n')
                if is_end == 1:
                    end = time.time()

                    with open(file_name_infor,'a',encoding='utf-8') as f:
                        f.write(str(per_num))
                        f.write('   ')
                        f.write(str(end - begin))
                        f.write('\r\n')
                    print('attack success and the pertubation is:',per_num)
                    break
            if is_end == 0:
                end = time.time()
                with open(file_name_infor,'a',encoding='utf-8') as f:
                    f.write(str(per_num))
                    f.write('   ')
                    f.write(str(y_pred))
                    f.write('   ')
                    f.write(str(end - begin))
                    f.write('\r\n')
                print('attack fail and the pertubation is:',per_num)

            # else:

                # input()


        



# if __name__ == '__main__':

#     dqn = DQN()
#     print("\nCollecting experience...")
#     ep_r = 0
#     is_end = 1
#     env.reset()
#     for i_episode in range(100000):
#         if is_end == 1:
#             env.reset()
#             is_end = 0
#             s = env.state
#             a = dqn.choose_action(s)
#         else:
#             s = s_
#             a = dqn.choose_action(s)  

#         a = dqn.choose_action(s)
#         x = torch.unsqueeze(torch.FloatTensor(s), 0)


#         r_1,r_2,s_,is_end = env.getReward(a, s)


#         # store the transitions of states
#         dqn.store_transition(s, a, r_1,r_2, s_)

#         ep_r += r_1

#         if dqn.memory_counter > MEMORY_CAPACITY:
#             dqn.learn()

#         if(i_episode % 100 == 0):
#             print(s)
#             print(a)
#             print(r_1,r_2)
#             print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
