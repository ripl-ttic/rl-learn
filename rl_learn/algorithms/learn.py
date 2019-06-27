import gin, os, time
import numpy as np
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_learn.modules import LEARN
from rl_learn.util import Data, get_batch_lang_lengths


@gin.configurable(blacklist=['lang_enc'])
class RunLEARN(object):
    def __init__(self,
        logdir,
        lang_enc,
        actions_file,
        data_file,
        lr=1e-4,
        vocab_size=296,
        n_actions=18,
        epochs=50,
        batch_size=32,
        gpu=True
    ):
        self.logdir = 'train/logs/learn/{}/'.format(lang_enc)
        self.data = Data(actions_file, data_file, lang_enc, n_actions)

        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.net = LEARN(vocab_size, n_actions, lang_enc)
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.net.parameters(), self.lr)
        self.lr = lr
        self.scheduler = optim.ExponentialLR(self.opt, 0.95)


        self.lang_enc = lang_enc

        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch = 0
        self.global_step = 0

    def save(self):
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'opt_state_dict': self.opt.state_dict()
        }, self.logdir + 'net.pkl')

    def train(self):
        if self.epochs and self.epoch > self.epochs:
            return

        self.best_val_acc = 0.0
        self.acc_val = 0.0
        try:
            while True:
                if self.epochs and self.epoch >= self.epochs:
                    break
                self.step()
                if self.acc_val > self.best_val_acc:
                    self.best_val_acc = self.acc_val
                    self.save()
        except KeyboardInterrupt:
            log("Caught Ctrl-C. Saving model and exiting...")


    def step(self):
        shuffle(self.data.train_data)
        acc_train, loss_train = self.run_epoch(self.data.train_data, training=1)
        acc_val, loss_val = self.run_epoch(self.data.valid_data, training=0)
        self.epoch += 1
        self.log_stats(acc_train, loss_train, acc_val, loss_val)
        self.acc_val = acc_val

    def run_epoch(self, data, training):
        start = 0
        loss = 0
        labels = []
        pred = []

        while start < len(data):
            batch_data = data[start:start+self.batch_size]
            batch_pred, batch_loss, batch_labels = self.run_batch(batch_data, training)

            start += self.batch_size
            loss += batch_loss    
            pred += list(batch_pred)
            labels += list(batch_labels)

            self.scheduler.step()

        correct = np.sum([1.0 if x == y else 0.0 for (x, y) in zip(pred, labels)])
        return correct / len(data), loss / len(data)

    def run_batch(self, batch_data, training):
        actions, langs, labels = zip(*batch_data)
        langs = np.asarray(langs)
        langs, lengths = get_batch_lang_lengths(langs, self.lang_enc)
        actions = torch.FloatTensor(actions)
        langs, lengths = torch.FloatTensor(langs), torch.LongTensor(lengths)
        labels = torch.LongTensor(labels)
        actions, langs, lengths, labels = actions.to(self.device), langs.to(self.device), lengths.to(self.device), labels.to(self.device) 
        if training == 1:
            self.opt.zero_grad()
            # lr = self.lr * (0.95 ** (self.global_step // 10000))
            # for param_group in self.opt.param_groups:
            #     param_group['lr'] = lr
            logits = self.net(actions, langs, lengths)
            loss = self.criterion(logits, labels)
            pred = logits.argmax(dim=1)

            loss.backward()
            self.opt.step()

            self.global_step += 1

        else:
            self.net.train(False)
            logits = self.net(actions, langs, lengths)
            loss = self.criterion(logits, labels)
            pred = logits.argmax(dim=1)
            self.net.train(True)
            
        return pred, loss, labels

    def log_stats(self, acc_train, loss_train, acc_val, loss_val):
        log("========================|  Epoch: {}  |========================".format(self.epoch), self.logdir)
        log('Train acc: {}'.format(acc_train), self.logdir)
        log('Train loss: {}'.format(loss_train), self.logdir)
        log('Valid acc: {}'.format(acc_val), self.logdir)
        log('Valid loss: {}'.format(loss_val), self.logdir)

def log(str, logdir):
    print(str)
    directory = logdir + 'log.txt'
    with open(directory, 'r') as f:
        f.write(str)
    return 