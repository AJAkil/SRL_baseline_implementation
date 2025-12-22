import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
import pickle
#A: 2,6
#B: 6,12,18
path_list_APIGraph = ['APIGraph/method_cluster_mapping_2000.pkl',
             'APIGraph/entities.txt',
             ]

for i in range(len(path_list_APIGraph)):
    if not os.path.exists(path_list_APIGraph[i]):
        path_list_APIGraph[i] = 'MaMaDroid/' + path_list_APIGraph[i]
#
with open(path_list_APIGraph[0], 'rb') as f:
    entity_embedding = pickle.load(f)
entities = []
for line in open(path_list_APIGraph[1],'r', encoding='utf-8'):
    entities.append(line.strip().split(',')[0])

class GNNGraph(object):
    def __init__(self, g, function_calls,call_times,label=0, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        # print(g.edges)
        # print(function_calls)
        # input()
        self.call_times = call_times
        self.num_nodes = len(g.nodes)
        self.node_tags = node_tags
        self.function_calls = function_calls
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())
        # print(g.edges)

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        # print(self.edge_pairs)
        # input()        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(edge_features.values()[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


class freq():
    def __init__(self,mama_lidu):
        self.req = np.zeros(2)
        self.gr = np.zeros(4)
        self.state = np.zeros(6)
        self.act = np.zeros(2)
        #self.inf = np.mat([[0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0]])
        self.world_bag = []
        self.nops = []




        if mama_lidu == 'family':

            # self.action_num = 7
            # self.action_dict = {0: 5, 1: 4, 2: 3, 3: 8, 4: 7, 5: 9, 6: 10}
            for line in open('data/nops_f2.txt', 'r', encoding='utf-8'):
                self.nops.append(line.strip().replace('.', '/'))

            self.action_num = len(self.nops)
            # print(self.nops)
            # print(self.action_num)
            # input()
            for line in open('data/Families.txt', 'r', encoding='utf-8'):
                self.world_bag.append(line.strip().replace('.', '/'))
            self.world_bag.append('self-defined')
            self.world_bag.append('obfuscated')
        elif mama_lidu == 'package':

            # self.action_num = 7
            # self.action_dict = {0: 5, 1: 4, 2: 3, 3: 8, 4: 7, 5: 9, 6: 10}
            for line in open('data/nops_f2.txt', 'r', encoding='utf-8'):
                self.nops.append(line.strip().replace('.', '/'))
            self.action_num = len(self.nops)

            for line in open('data/Packages.txt', 'r', encoding='utf-8'):
                self.world_bag.append(line.strip())
            allpacks=[]
            for i in self.world_bag:
                #allpacks.append(i.split('.')[1:]) # why do not need pos 1
                allpacks.append(i.split('.')[:])
            self.pos_p=[[],[],[],[],[],[],[],[],[]]
            for i in allpacks:
                k=len(i)
                for j in range(0,k):
                    if i[j] not in self.pos_p[j]:#add all name in pos j
                        self.pos_p[j].append(i[j])
            # print(self.pos_p)
            # input()
            self.world_bag.append('self-defined')
            self.world_bag.append('obfuscated')            
        elif mama_lidu == 'class':
            for line in open('data/Classes.txt', 'r', encoding='utf-8'):
                self.world_bag.append(line.strip().replace('.', '/'))
            self.world_bag.append('self-defined')
            self.world_bag.append('obfuscated')

    def PackAbs(self,call, pos):
        '''
        MaMaDroid Package
        '''
        partitions = call[1:].split(';->')[0].split('/')
        package = ""
        for i in range(0, len(partitions)):
            if partitions[i] in pos[i]:
                package = package + partitions[i] + '.'
            else:
                if package == "" or package == 'com.':
                    package = None
                else:
                    if package.endswith('.'):
                        package = package[0:-1]
                break
        if package not in self.world_bag and package != None:
            partitions = package.split('.')
            for i in range(0, len(partitions)):  # 4
                package = package[0:-(len(partitions[len(partitions) - i - 1]) + 1)]
                if package in self.world_bag:
                    return package
        if package == "" or package == 'com.':
            package = None
        return package

    def get_package_call_times_from_function_calls(self,function_calls):
        g = nx.Graph()
        for i in range(len(self.world_bag)):
            g.add_node(i)
        package_call_times = np.zeros((len(self.world_bag), len(self.world_bag)))
        for cur_pair in function_calls:
            cur_pair = cur_pair.split(' ')
            temp = cur_pair[0]
            # print(temp)
            # input()
            match=self.PackAbs(temp,self.pos_p)

            # print(temp,self.pos_p,match)
            # input()
            if match == None:
                splitted = temp[1:].split(';->')[0].split('/')
                obfcount=0
                for k in range (0,len(splitted)):
                    if len(splitted[k])<3:
                        obfcount+=1
                if obfcount>=len(splitted)/2:
                    match='obfuscated'
                else:
                    match='self-defined'
            caller = self.world_bag.index(match)

            # print(cur_pair[0], match, caller)  
            ###
            temp = cur_pair[2]
            match=self.PackAbs(temp, self.pos_p)
            # print(temp,match)
            if match == None:
                splitted = temp[1:].split(';->')[0].split('/')
                obfcount=0
                for k in range (0,len(splitted)):
                    if len(splitted[k])<3:
                        obfcount+=1
                if obfcount>=len(splitted)/2:
                    match='obfuscated'
                else:
                    match='self-defined'
            
            callee = self.world_bag.index(match)
            # print(cur_pair[2], match, callee)
            # input()
            package_call_times[caller][callee] += 1
            g.add_edge(caller, callee)
        # print(package_call_times)
        call_times = sp.coo_matrix(package_call_times)
        # print(call_times)
        # input()
        for i in range(len(self.world_bag)):
            sumer = sum(package_call_times[i])
            if sumer == 0:
                continue
            for j in range(len(self.world_bag)):
                package_call_times[i][j] = package_call_times[i][j]/sumer
        node_feature = []
        for i in g.nodes:
            tem = np.zeros(package_call_times.shape[0])
            tem[i] = 1
            tem = tem*(np.sum(package_call_times[i,:])+np.sum(package_call_times[:,i]) - package_call_times[i,i])
            node_feature.append(tem)
        node_feature = np.array(node_feature)

        state = GNNGraph(g,function_calls,package_call_times,node_features=node_feature)
        return state


    def get_class_call_times_from_function_calls(self,function_calls):
        g = nx.Graph()
        class_call_times = np.zeros((len(self.world_bag), len(self.world_bag)))
        for cur_pair in function_calls:
            function_pair = cur_pair.split(' ')
            print(cur_pair)
            print(function_pair)

            temp = function_pair[0]
            caller_c = temp[1:].split(';->')[0]
            match = None
            if caller_c in self.world_bag:
                match = caller_c
            if match == None:
                splitted = caller_c.split('/')
                obfcount=0
                for k in range (0,len(splitted)):
                    if len(splitted[k])<3:
                        obfcount+=1
                if obfcount>=len(splitted)/2:
                    match='obfuscated'
                else:
                    match='self-defined'

            # print(function_pair[0], match)
            caller = self.world_bag.index(match)
            # print(function_pair[0], match, caller)  
            ###
            temp = function_pair[2]
            callee_c = temp[1:].split(';->')[0]
            match = None
            # print(callee_c)
            if callee_c in self.world_bag:
                # print(1111111)
                match = callee_c
            if match == None:
                splitted = callee_c.split('/')
                obfcount=0
                for k in range (0,len(splitted)):
                    if len(splitted[k])<3:
                        obfcount+=1
                if obfcount>=len(splitted)/2:
                    match='obfuscated'
                else:
                    match='self-defined'    
            callee = self.world_bag.index(match)
            # print(cur_pair[2], match, callee)
            # input()
            class_call_times[caller][callee] += 1
            g.add_node(caller)
            g.add_node(callee)
            g.add_edge(caller, callee)
            # print(cur_pair, caller, callee)

        state = GNNGraph(g,function_calls)
        return state

    def get_family_call_times_from_function_calls(self,function_calls):
        g = nx.Graph()
        for i in range(len(self.world_bag)):
            g.add_node(i)
        family_call_times = np.zeros((len(self.world_bag), len(self.world_bag)))
        for cur_pair in function_calls:

            cur_pair = cur_pair.split(' ')
            temp1 = cur_pair[0][1:]
            for family in self.world_bag:
                # print(family)
                if temp1.startswith(family):
                # if family in cur_pair[0]:
                    temp1 = family
                    break

            temp2 = cur_pair[2][1:]


            for family in self.world_bag:
                if family in temp2:
                    temp2 = family
                    break


            if temp1 not in self.world_bag:
                # print('1111',temp1)
                splitted = temp1.split(';->')[0].split('/')
                obfcount=0
                for k in range (0,len(splitted)):
                    if len(splitted[k])<3:
                        obfcount += 1
                if obfcount >= len(splitted)/2:
                    temp1 = 'obfuscated'
                else:
                    temp1 = 'self-defined'
            # print('1111',splitted)
            caller = self.world_bag.index(temp1)
            if temp2 not in self.world_bag:
                splitted = temp2.split(';->')[0].split('/')
                obfcount=0
                for k in range (0,len(splitted)):
                    if len(splitted[k])<3:
                        obfcount += 1
                if obfcount >= len(splitted)/2:
                    temp2 ='obfuscated'
                else:
                    temp2 ='self-defined'
            
            callee = self.world_bag.index(temp2)
            family_call_times[caller][callee] += 1
            # g.add_node(caller)
            # g.add_node(callee)
            g.add_edge(caller, callee)
        # family_call_times = sp.coo_matrix(family_call_times)
        # print(family_call_times)
        # input()
        for i in range(len(self.world_bag)):
            sumer = sum(family_call_times[i])
            if sumer == 0:
                continue
            for j in range(len(self.world_bag)):
                family_call_times[i][j] = family_call_times[i][j]/sumer




        # input()
        node_feature = []
        for i in g.nodes:
            tem = np.zeros(family_call_times.shape[0])
            tem[i] = 1
            tem = tem*(np.sum(family_call_times[i,:])+np.sum(family_call_times[:,i]) - family_call_times[i,i])
            node_feature.append(tem)
        node_feature = np.array(node_feature)

        state = GNNGraph(g,function_calls,family_call_times,node_features=node_feature)
        # input()
        return state


    def data2state(self,data,mama_lidu):
        if mama_lidu == 'family':
            state = self.get_family_call_times_from_function_calls(data)
        elif mama_lidu == 'package':
            state = self.get_package_call_times_from_function_calls(data)
        elif mama_lidu == 'class':
            state = self.get_class_call_times_from_function_calls(data)
            # state = GNNGraph(g, l, node_tags, node_features)
        return state

    def feature_trans_APIGraph(self,func_calls_org):
        '''
        params：
            all_funcs_org: 原始的函数列表
            func_calls_org: 原始的函数调用对列表（index）
        return:
            all_funcs_APIGraph: APIGraph增强后的函数列表
            func_calls_APIGraph：相应的函数调用对列表 （index）
            map2APIGraph：原始函数列表index到APIGraph增强后的index映射表
        '''
        # all_funcs_APIGraph = []
        # all_funcs_dict = {}
        func_calls_APIGraph = []
        data = []
        # for func in all_funcs_org:
        #     func0 = func.split('(')[0]
        #     func = func.split('(')[0][1:].replace('/', '.').replace(';->', '.').replace('$', '.').replace('<init>',
        #                                                                                                           'init')
        #     if func in entity_embedding.keys():
        #         func = entities[entity_embedding[func]]

        #         out1 = func[:func.rfind('.')]
        #         out2 = func[func.rfind('.') + 1:]
        #         if out2 == 'init':
        #             out2 = '<init>'
        #         func = out1 + ';->' + out2
        #         func = 'L' + func.replace('.', '/')
        #         # 输出形式与输入保持一致
        #     else:
        #         func = func0
        #     if func not in all_funcs_dict.keys():
        #         all_funcs_APIGraph.append(func)
        #         all_funcs_dict[func] = 1
        # print('all_funcs_APIGraph', all_funcs_APIGraph)#Lcom/qihoo/util/Configuration;-><init>()V
        # print(entity_embedding.keys())
        for func_call in func_calls_org:
            caller0 = func_call.split(' ')[0].split('(')[0] # index
            callee0 = func_call.split(' ')[-1].split('(')[0]
            # caller0 = all_funcs_org[caller0].split('(')[0]
            # callee0 = all_funcs_org[callee0].split('(')[0]
            # print(caller0)
            # print(callee0)
            # input()
            caller = caller0[1:].replace('/', '.').replace(';->', '.').replace('$', '.').replace('<init>', 'init')
            callee = callee0[1:].replace('/', '.').replace(';->', '.').replace('$', '.').replace('<init>', 'init')
            # print(caller)
            # print(callee)
            # input()

            # print(caller, callee)
            if caller in entity_embedding.keys():
                caller = entities[entity_embedding[caller]]
                out1 = caller[:caller.rfind('.')]
                out2 = caller[caller.rfind('.') + 1:]
                if out2 == 'init':
                    out2 = '<init>'
                caller = out1 + ';->' + out2
                caller = 'L' + caller.replace('.', '/')
                # print(caller)
            else:
                caller = caller0
            if callee in entity_embedding.keys():
                callee = entities[entity_embedding[callee]]
                out1 = callee[:callee.rfind('.')]
                out2 = callee[callee.rfind('.') + 1:]
                if out2 == 'init':
                    out2 = '<init>'
                callee = out1 + ';->' + out2
                callee = 'L' + callee.replace('.', '/')
            else:
                callee = callee0
            # caller_index = all_funcs_APIGraph.index(caller)
            # callee_index = all_funcs_APIGraph.index(callee)
            data_tem = caller + ' invoke ' + callee
            data.append(data_tem)
            # print(data_tem)
            # print(callee)
            # input()
            # func_calls_APIGraph.append(str(caller_index) + ' ' + func_call.split(' ')[1] +' ' + str(callee_index) + '\n')

      

        return data


    def data2state_APTGRAPH(self,data,mama_lidu):
        if mama_lidu == 'family':
            data_api = self.feature_trans_APIGraph(data)
            state = self.get_family_call_times_from_function_calls(data_api)
        elif mama_lidu == 'package':
            data_api = self.feature_trans_APIGraph(data)
            state = self.get_package_call_times_from_function_calls(data_api)
        elif mama_lidu == 'class':
            data_api = self.feature_trans_APIGraph(data)
            state = self.get_class_call_times_from_function_calls(data_api)
            # state = GNNGraph(g, l, node_tags, node_features)
        return state




    def reset(self,data,mama_lidu,classifier,clf_type):
        if "_APIGraph" in clf_type:
            self.state = self.data2state_APTGRAPH(data,mama_lidu)

            input_pca = np.reshape(self.state.call_times,(-1,int(len(self.world_bag)*len(self.world_bag))))
            if mama_lidu != "class":
                x_test = classifier.model_pca.transform(input_pca)
            y_pred = classifier.test(x_test)
            self.init_pro = y_pred
            self.init_edge_num = len(self.state.function_calls)
        else:
            self.state = self.data2state(data,mama_lidu)

            input_pca = np.reshape(self.state.call_times,(-1,int(len(self.world_bag)*len(self.world_bag))))
            if mama_lidu != "class":
                x_test = classifier.model_pca.transform(input_pca)
            y_pred = classifier.test(x_test)
            self.init_pro = y_pred
            self.init_edge_num = len(self.state.function_calls)            
        # print(self.state)
        # input()
        # return state


    def getReward(self,action,data,mama_lidu,classifier,attack_method,clf_type):
        action = np.array(action)
        # print(np.argmax(action),int(self.action_dict[np.argmax(action)]),"!!!!!!!!!!!!!!!!!")
        # print(action,np.argmax(action))
        inset_calls  = "Lcom1111/qihoo/util/Configuration;-><init>()V" + " invoke-direct "+ self.nops[np.argmax(action)]   
        data = list(data)

        data.append(inset_calls)

        data = np.array(data)
        if "_APIGraph" in clf_type:        
            new_state = self.data2state_APTGRAPH(data,mama_lidu)
        else:
            new_state = self.data2state(data,mama_lidu)
            

        input_pca = np.reshape(new_state.call_times,(-1,int(len(self.world_bag)*len(self.world_bag))))

        if mama_lidu != "class":
            x_test = classifier.model_pca.transform(input_pca)
        # print(x_test)
        # input()
        y_pred = classifier.test(x_test)
        # print(self.init_pro,y_pred)
        # print(self.init_edge_num,len(new_state.function_calls))
        r = 0
        is_end = 0
        # print(self.init_pro[0][0][0])
        # print(y_pred[0][0][0])
     
        # input()
        if attack_method == 'SRL_no':
            if y_pred[0][0][0] < 0.5:
                r = 1
        elif attack_method == 'SRL':
            if self.init_pro[0][0][0]>y_pred[0][0][0]:
                r = 1
        elif attack_method == 'SRI':
            if self.init_pro[0][0][0]>y_pred[0][0][0]:
                r = 1
        if y_pred[0][0][0] < 0.5:
            is_end = 1       
        per_num = len(new_state.function_calls) - self.init_edge_num
        return data,new_state,is_end,r,y_pred,per_num 


        # #A和B机器分配完成
        # if(e1[0] + e1[1] == self.state[0] and e2[0] + e2[1] + e2[2] == self.state[1]):
        #     r =  5

        # #判断是否与外界干扰冲突    
        # for i in range(0,len(self.gr)):
        #     if(action[i] == 1 and self.state[i+2] == 1):
        #         r = r - 2

        # #判断是否与自身冲突
        # # if(e1[0] == 1 and e1[2] == 1):
        # #     r = r - 1

        # #需要分配且不分配
        # if(e1[0]+e1[1] < self.state[0]):
        #     r = r - 1
        # if(e2[0]+e2[1]+e2[2] < self.state[1]):
        #     r = r - 1 
        # #不需要分配且分配    
        # if(e1[0]+e1[1] > self.state[0]):
        #     r = r - 0.5
        # if(e2[0]+e2[1]+e2[2] > self.state[1]):
        #     r = r - 0.5
        # # if(e1[0] == 1 and e1[2] == 1):
        # #     r = r - 1
        # # if(e2[0] == 1 and e2[1] == 1):
        # #     r = r - 1
        # # if(e1[0]+e1[1]+e1[2] < self.req[0]):
        # #     r = r - self.req[0] * 2  + e1[0]+e1[1]+e1[2]
        # # if(e2[0]+e2[1]+e2[2] < self.req[1]):
        # #     r = r - self.req[1] * 2  + e2[0]+e2[1]+e2[2]        
        # # for i in range(0,len(self.gr)):
        # #     if(action[i] == 1 and self.gr[i] == 1):
        # #         r = r - 0.5
        # return r

#print(env.state)