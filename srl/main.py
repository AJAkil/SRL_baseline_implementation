
import classifier_MAMA
import os
import shutil
import sys
import argparse
import dqn_f
sys.path.append("classifier")
import RF_demo
import dnn
import cnn
import AdaBoost
import K_1NN_demo
import K_3NN_demo
import joblib
class Classifier( ):
    def __init__(self, granularity, clf_type):
        self.clf_type = clf_type
        self.model_pca = None
        self.granularity = granularity
        print(self.clf_type, self.granularity)
        if not os.path.exists('MaMaDroid/MaMaData/keras_%s' % (self.granularity)):
            os.makedirs('MaMaDroid/MaMaData/keras_%s' % (self.granularity))
        if self.clf_type.endswith('APIGraph'):
            print('####### APIGraph ################')
            self.APIGraph = '_APIGraph'
        else:
            self.APIGraph = ''  
    def model_load(self):        
        # self.model_pca = joblib.load('keras_%s/pca%s.m'%(self.granularity, self.APIGraph)) # 读入模型pca.m
        APIGraph = False
        clf_type = self.clf_type.replace("_APIGraph", "")
        if "_APIGraph" in self.clf_type:
            APIGraph = True
        if clf_type =="dnn":
            self.clf  = dnn.model_load(self.granularity, APIGraph = APIGraph)

        elif clf_type =="cnn":
            self.clf  = cnn.model_load(self.granularity, APIGraph = APIGraph)
        elif clf_type =="RF":
            self.clf  = RF_demo.model_load(self.granularity, APIGraph = APIGraph)
        elif clf_type =="ab":
            self.clf  = AdaBoost.model_load(self.granularity, APIGraph = APIGraph)
        elif clf_type =="1_NN":
            self.clf  = K_1NN_demo.model_load(self.granularity, APIGraph = APIGraph)
        elif clf_type =="3_NN":
            self.clf  = K_3NN_demo.model_load(self.granularity, APIGraph = APIGraph)

        else:
            print(" error type of classifier ", self.clf_type)
            input()
    def load_PCA(self):
        pca_model_path = 'mamadata/keras_%s/pca%s.m' % (self.granularity, self.APIGraph)
        if os.path.exists(pca_model_path):
            self.model_pca = joblib.load(pca_model_path)
        else:
            print('absence of pca model')
            input()
    def train_PCA(self, dataset):
        print('############## PCA training ###################')
        if self.granularity == "family":
            PCA_n_components = 100
        elif self.granularity == "package":
            PCA_n_components = 300
        else:
            PCA_n_components = 300
        pca_model_path = 'MaMaDroid/MaMaData/keras_%s/pca%s.m' % (self.granularity, self.APIGraph)

        print('PCA_n_components', PCA_n_components)
        if self.granularity != "class":
            data_i_numpy = []
            for item in dataset:
                data_i_numpy.append(item[self.granularity + '_call_prob'].todense().reshape(-1))
            data_i_numpy = np.array(data_i_numpy)
            self.model_pca = PCA(n_components=PCA_n_components)  # n_components设置降维后的维度
            data_i_numpy = data_i_numpy.reshape((data_i_numpy.shape[0], data_i_numpy.shape[2]))
            print(data_i_numpy.shape)
            self.model_pca.fit(data_i_numpy)

        elif self.granularity == "class":
            self.model_pca = IncrementalPCA(n_components=100, batch_size=10)
            num = 100
            for batch in range(100):
                data_part = []
                print(batch)
                for i in range(num):
                    data_i_numpy = dataset[num * batch + i]['class_call_prob'].todense()  # or family_call_prob
                    data_i_numpy = data_i_numpy.reshape(-1)
                    data_part.append(data_i_numpy)
                data_part = np.array(data_part)
                data_part = data_part.reshape(data_part.shape[0], data_part.shape[2])
                self.model_pca.partial_fit(data_part)

        joblib.dump(self.model_pca, pca_model_path)  # 将模型保存到pca.m文件中
    def train(self, x, y):
        APIGraph = False;
        clf_type = self.clf_type.replace("_APIGraph", "")
        if "_APIGraph" in self.clf_type:
            APIGraph = True
        print("#####classifier#####")
        # if self.clf_type == "svm":
        #     self.clf = svm_demo_APP.train(x, y, self.granularity)
        if clf_type =="dnn":
            self.clf  = dnn.train(x, y, self.granularity, APIGraph = APIGraph)
        elif clf_type =="cnn":
            self.clf  = cnn.train(x, y, self.granularity, APIGraph = APIGraph)
        elif clf_type =="RF":
            self.clf  = RF_demo.train(x, y, self.granularity, APIGraph = APIGraph)
        elif clf_type =="ab":
            self.clf  = AdaBoost.train(x, y, self.granularity, APIGraph = APIGraph)
        elif clf_type =="1_NN":
            self.clf  = K_1NN_demo.train(x, y, self.granularity, APIGraph = APIGraph)
        elif clf_type =="3_NN":
            self.clf  = K_3NN_demo.train(x, y, self.granularity, APIGraph = APIGraph)

        else:
            print(" error type of classifier ")

    def get_grad(self, x):
        return dnn.get_grad(self.clf, x)
    def test(self, x):
        clf_type = self.clf_type.replace("_APIGraph", "")
        # print("#####test#####")
        # if self.clf_type == "svm":
        #     y = svm_demo_APP.test(self.clf, x)
        if clf_type =="dnn":
            y = dnn.test(self.clf, x)
        elif clf_type =="cnn":
            y = cnn.test(self.clf, x)
        elif clf_type =="RF":
            y = RF_demo.test(self.clf, x)
        elif clf_type =="ab":
            y = AdaBoost.test(self.clf, x)
        elif clf_type =="1_NN":
            y = K_1NN_demo.test(self.clf, x)    
        elif clf_type =="3_NN":
            y = K_3NN_demo.test(self.clf, x)

        else:
            print(" error type of classifier ")
            return 0
        return y

# def CLF(CLF_TYPE, MAMA_LIDU):

#     ## 
#     print('############### Load the target classifier model ###############')
#     print("The classifier is ",CLF_TYPE,"The granularity is ",MAMA_LIDU)

#     classifier = classifier_MAMA.Classifier( MAMA_LIDU,CLF_TYPE) 
#     classifier.model_load()
#     return classifier

#     def load_PCA(self):
#         pca_model_path = 'MaMaDroid/MaMaData/keras_%s/pca%s.m' % (self.granularity, self.APIGraph)
#         if os.path.exists(pca_model_path):
#             self.model_pca = joblib.load(pca_model_path)
#         else:
#             print('absence of pca model')

def get_data(file_path,granularity):
    file_list = os.listdir(file_path)
    data = []
    for file in file_list:
        all_funcs = []
        function_calls = []


        data_path = os.path.join(file_path,file)
        function_call_path = data_path + '/func_calls.txt'
        all_funcs_path = data_path + '/all_functions.txt'


        if not os.path.exists(function_call_path):
            print(function_call_path)
            continue

        for line in open(all_funcs_path, 'r', encoding='utf-8'):
            all_funcs.append(line.strip())
        for line in open(function_call_path, 'r', encoding='utf-8'):
            function_calls.append(line.strip())
        count = 0
        for i_pair in function_calls:
            caller = int(i_pair.split(' ')[0])
            callee = int(i_pair.split(' ')[2])
            function_calls[count] = all_funcs[caller] + ' ' + i_pair.split(' ')[1] + ' ' + all_funcs[callee]
            count += 1
        data.append(function_calls)
    return data



if __name__ == '__main__':

    MAMA_LIDU_list = [ 'family' ]  #[]
    CLF_TYPE_list = ['dnn']
    attack_method= ["SRL"] #['SRL','SRL_no',"SRI","SRI_no"]
    # APIGraph
    for clf_type in CLF_TYPE_list:
        for mama_lidu in MAMA_LIDU_list:
            for attack in attack_method:
                # if "_APIGraph" in clf_type:
                #     API_Graph_enhanced = 1 
                # else:
                #     API_Graph_enhanced = 0
                # args.CLF_TYPE, args.MAMA_LIDU = clf_type, mama_lidu
                CLF = Classifier(mama_lidu,clf_type)
                CLF.load_PCA()
                CLF.model_load()
                file_path = os.path.join('data','attack_data_test',mama_lidu+'_'+clf_type)
                file_list = os.listdir(file_path)
                dataset =  get_data(file_path,mama_lidu)
                # input()
                count = 0
                for data,file in zip(dataset, file_list):
                    count +=1

                    # if file =='0069B6C582D9EF3FDFF5C09E7E4F0C097399529A06625BCC2920BA3C488582BD':
                    if count >= 0:

                        print("Now , processing the sample  id", count)
                        file = os.path.join('data','attack_data_test',mama_lidu+'_'+clf_type,file)
                        print(file)

                        dqn_f.test(CLF,data,clf_type, mama_lidu,file,attack)