import models, torch
from copy import deepcopy
from sklearn.metrics import f1_score
import math

class Server(object):

    def __init__(self, conf, test_dataset, val_dataset, n_input):

        self.conf = conf

        self.global_model = models.MLP(n_input, 512, conf["num_class"])

        self.eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.conf["batch_size"], shuffle=True)

        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def model_aggregate(self, clients_model, weights):

        new_model = {}

        for name, params in self.global_model.state_dict().items():
            new_model[name] = torch.zeros_like(params)

        for key in clients_model.keys():

            for name, param in clients_model[key].items():
                new_model[name]= new_model[name] + clients_model[key][name] * weights[key]

        self.global_model.load_state_dict(new_model)

    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        predict = []
        label = []
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            predict.extend(pred.numpy())
            label.extend(target.numpy())

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        f1 = f1_score(predict, label)

        return f1, acc, total_l

    def cal_ls(self, model):
        self.global_model.load_state_dict(model)
        self.global_model.eval()

        total_loss = 0.0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,reduction='sum').item()

        total_l = total_loss / dataset_size

        return total_l

    def cal_credibility(self, e,a):
        total_e = 0
        cred = {}
        for key in e.keys():
            total_e += a*math.exp(e[key])

        for key in e.keys():
            cred[key] = 1 - (a * math.exp(e[key]) / total_e)
        return cred

    def update_weights(self,num, cred):
        total_w = 0
        new_weight = {}
        for key in num.keys():
            total_w += num[key] * cred[key]

        for key in num.keys():
            new_weight[key] = (num[key] * cred[key]) / total_w
        return new_weight


