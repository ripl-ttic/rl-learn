import gin, os, time
import numpy as np
from random import shuffle
from dl.util import Checkpointer, logger, rng

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_learn.modules import LEARN
from rl_learn.util import Data, get_batch_lang_lengths

@gin.configurable(blacklist=['logdir'])
class RunLearn(object):
    def __init__(self,
        logdir,
        actions_file,
        data_file,
        lr=0.0001,
        lang_enc='onehot',
        vocab_size=296,
        n_actions=18,
        epochs=50,
        batch_size=32,
        gpu=True
    ):
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(self.logdir, 'ckpts'))
        self.data = Data(actions_file, data_file, lang_enc, n_actions)

        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.net = LEARN(vocab_size, n_actions, lang_enc)
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.net.parameters())
        self.lr = lr

        logger.configure(logdir, ['stdout', 'log'])

        self.lang_enc = lang_enc

        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch = 0
        self.global_step = 0

    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            'epoch': self.epoch
        }

    def load_state_dict(self):
        self.net.load_state_dict(state_dict['net'])
        self.opt.load_state_dict(state_dict['opt'])
        self.epoch = state_dict['epoch']

    def save(self):
        self.ckptr.save(self.state_dict(), self.epoch)

    def load(self, epoch=None):
        self.load_state_dict(self.ckptr.load(epoch))

    def train(self):
        config = gin.operative_config_str()
        logger.log("=================== CONFIG ===================")
        logger.log(config)
        with open(os.path.join(self.logdir, 'config.gin'), 'w') as f:
            f.write(config)
        if len(self.ckptr.ckpts()) > 0:
            self.load()
        if self.epoch == 0:
            cstr = config.replace('\n', '  \n')
            cstr = cstr.replace('#', '\\#')
            logger.add_text('config', cstr, 0, time.time())
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
                    self.save()
        except KeyboardInterrupt:
            logger.log("Caught Ctrl-C. Saving model and exiting...")
        if self.epoch not in self.ckptr.ckpts():
            self.save()

        logger.export_scalars(self.ckptr.format.format(self.epoch) + '.json')
        logger.reset()

    def step(self):
        shuffle(self.data.train_data)
        acc_train, loss_train = self.run_epoch(self.data.train_data, training=1)
        acc_val, loss_val = self.run_epoch(self.data.valid_data, training=0)
        self.epoch += 1
        self.log(acc_train, loss_train, acc_val, loss_val)
        self.acc_val = acc_val

    def run_epoch(self, data, training):
        start = 0
        loss = 0
        labels = []
        pred = []

        while start < len(data):
            batch_data = data[start:start+self.batch_size]
            batch_pred, batch_loss = self.run_batch(batch_data, training)

            start += self.batch_size
            loss += batch_loss    
            pred += list(batch_pred)
            labels += list(batch_labels)

        correct = np.sum([1.0 if x == y else 0.0 for (x, y) in zip(pred, labels)])
        return correct / len(data), loss / len(data)

    def run_batch(self, batch_data, training):
        self.opt.zero_grad()
        lr = self.lr * 0.95 ** (self.global_step // 10000)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

        actions, langs, labels = zip(*batch_data)
        langs = np.array(langs)
        langs, lengths = get_batch_lang_lengths(langs, self.lang_enc)
        actions = torch.FloatTensor(actions)
        langs, lengths = torch.from_numpy(langs), torch.from_numpy(lengths).long()
        labels = torch.LongTensor(labels)
        actions, langs, lengths, labels = actions.to(self.device), langs.to(self.device), lengths.to(self.device), labels.to(self.device)
        if training:
            self.net.train()
            logits = self.net(actions, langs, labels)
            labels = F.one_hot(labels.long())
            loss = self.criterion(logits, labels)
            pred = logits.argmax(dim=1)

            loss.backward()
            self.opt.step()

            self.global_step += 1

        else:
            self.net.eval()
            logits = self.net(actions, langs, labels)
            labels = F.one_hot(labels.long())
            loss = self.criterion(logits, labels)
            pred = logits.argmax(dim=1)

        return pred, loss

    def log(self, acc_train, loss_train, acc_val, loss_val):
        logger.log("========================|  Epoch: {}  |========================".format(self.epoch))
        logger.logkv('Train acc', acc_train)
        logger.logkv('Train loss', loss_train)
        logger.logkv('Valid acc', acc_val)
        logger.logkv('Valid loss', loss_val)
        logger.dumpkvs()
