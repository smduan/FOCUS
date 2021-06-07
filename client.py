import models, torch


class Client(object):

    def __init__(self, conf, model, train_dataset, val_dataset):

        self.conf = conf

        self.local_model = model

        self.client_id = id

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf["batch_size"],shuffle=True)

    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        self.local_model.train()
        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data = data.view(data.size(0),-1)
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)

                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                optimizer.step()
            print("Epoch %d done." % e)

        return self.local_model.state_dict()

    def cal_ll(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.eval()
        total_loss = 0.0
        dataset_size = 0
        for batch_id, batch in enumerate(self.val_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
        total_l = total_loss/dataset_size
        return  total_l