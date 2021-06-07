import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden // 2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden // 2)

        self.hidden_3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden // 4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden // 8, n_output)  # output layer

    def forward(self, X):
        x = X.view(X.shape[0], -1)
        x = F.relu(self.hidden_1(x))  # hidden layer 1
        x = self.dropout(self.bn1(x))

        x = F.relu(self.hidden_2(x))  # hidden layer 2
        x = self.dropout(self.bn2(x))

        x = F.relu(self.hidden_3(x))  # hidden layer 3
        x = self.dropout(self.bn3(x))

        x = F.relu(self.hidden_4(x))  # hidden layer 4
        feature = self.dropout(self.bn4(x))

        x = self.out(feature)

        return x


def get_model(name="vgg16", pretrained=True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
    elif name =="mlp":
        model = MLP(155,512,2)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model