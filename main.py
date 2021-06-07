import argparse, json
import datetime
import os
import logging
import torch, random
import pandas as pd

from server import *
from client import *
from  datasets import MyDataset, get_dataset

conf_file = './utils/conf.json'
with open(conf_file, 'r') as f:
    conf = json.load(f)
log = './logs/'

# ##数据文件，无噪音
# train_dataset_file = {
#         "alice": './data/adult/clean/even_split/adult_part_0.csv',
#         "bob": './data/adult/clean/even_split/adult_part_1.csv',
#         "lace":'./data/adult/clean/even_split/adult_part_2.csv',
#         "laodou":'./data/adult/clean/even_split/adult_part_3.csv'
# }

# ##数据文件，单节点有噪音
# train_dataset_file = {
#         "alice": './data/adult/clean/even_split/adult_part_0.csv',
#         "bob": './data/adult/clean/even_split/adult_part_1.csv',
#         "lace":'./data/adult/clean/even_split/adult_part_2.csv',
#         "laodou":'./data/adult/symmetric_flipping/even_split/0.1/adult_part_3_0.1.csv'
# }

###数据文件，全部有噪音
train_dataset_file = {
        "alice": './data/adult/symmetric_flipping/even_split/0.1/adult_part_0_0.1.csv',
        "bob": './data/adult/symmetric_flipping/even_split/0.1/adult_part_1_0.1.csv',
        "lace":'./data/adult/symmetric_flipping/even_split/0.1/adult_part_2_0.1.csv',
        "laodou":'./data/adult/symmetric_flipping/even_split/0.1/adult_part_3_0.1.csv'
}

test_dataset_file = './data/adult/adult_test.csv'

train_datasets = {}
val_datasets = {}
##各节点数据量
number_samples = {}

##读取数据集,训练数据拆分成训练集和测试集
for key in train_dataset_file.keys():
    train_dataset = pd.read_csv(train_dataset_file[key])
    if "is_noise" in train_dataset.columns:
        train_dataset = train_dataset.drop(columns=['is_noise'])
    val_dataset = train_dataset[:int(len(train_dataset)*0.3)]
    train_dataset = train_dataset[int(len(train_dataset)*0.3):]
    train_datasets[key] = MyDataset(train_dataset)
    val_datasets[key] = MyDataset(val_dataset)

    number_samples[key] = len(train_dataset)

##测试集,在Server端测试模型效果，拆分成测试集和一个用于评估数据节点的数据集
test_dataset = pd.read_csv(test_dataset_file)
eval_dataset = test_dataset[:int(len(test_dataset)*0.3)]
test_dataset = test_dataset[int(len(test_dataset)*0.3):]
n_input = test_dataset.shape[1] - 1
eval_dataset = MyDataset(eval_dataset)
test_dataset = MyDataset(test_dataset)

###初始化每个节点聚合权值
client_weight = {}
if conf["is_init_avg"]:
    for key in number_samples.keys():
        client_weight[key] = 0.25
else:
    for key in number_samples.keys():
        total_sumples = sum(number_samples.values())
        client_weight[key] = number_samples[key] / total_sumples

print(client_weight)

##保存节点
clients = {}
#保存节点模型
clients_models = {}

if __name__ == '__main__':

    server = Server(conf, test_dataset, eval_dataset, n_input)

    ###更新权值
    ls_k ={}
    ll_k = {}
    e_k = {}

    for key in train_datasets.keys():
        clients[key] = Client(conf, server.global_model, train_datasets[key], val_datasets[key])

    for e in range(conf["global_epochs"]):

        for key in clients.keys():
            print('training {}...'.format(key))
            model_k = clients[key].local_train(server.global_model)
            ls_k[key] = server.cal_ls(model_k)
            clients_models[key] = model_k

        #加权聚合
        server.model_aggregate(clients_models, client_weight)

        #更新权值
        for key in clients.keys():
            ll_k[key] = clients[key].cal_ll(server.global_model)
            e_k[key] = ls_k[key] + ll_k[key]
        c_k = server.cal_credibility(e_k, 10)

        if not conf['is_init_avg']:
            client_weight = server.update_weights(number_samples, c_k)
        print(client_weight)

        f1, acc, loss = server.model_eval()

        print("Epoch %d, f1: %f, acc: %f, loss: %f\n" % (e, f1, acc, loss))

    torch.save(server.global_model.state_dict(), './save_model/adult-noise.pth')