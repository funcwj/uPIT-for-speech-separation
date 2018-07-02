#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import time

import torch as th
import torch.nn.functional as F

from itertools import permutations
from dataset import logger
from torch.nn.utils.rnn import PackedSequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def create_optimizer(optimizer, params, **kwargs):
    supported_optimizer = {
        'sgd': th.optim.SGD,  # momentum, weight_decay, lr
        'rmsprop': th.optim.RMSprop,  # momentum, weight_decay, lr
        'adam': th.optim.Adam,  # weight_decay, lr
        'adadelta': th.optim.Adadelta,  # weight_decay, lr
        'adagrad': th.optim.Adagrad,  # lr, lr_decay, weight_decay
        'adamax': th.optim.Adamax  # lr, weight_decay
        # ...
    }
    if optimizer not in supported_optimizer:
        raise ValueError('Now only support optimizer {}'.format(optimizer))
    if optimizer != 'sgd' and optimizer != 'rmsprop':
        del kwargs['momentum']
    opt = supported_optimizer[optimizer](params, **kwargs)
    logger.info('Create optimizer {}: {}'.format(optimizer, kwargs))
    return opt


def packed_sequence_cuda(packed_sequence):
    if not isinstance(packed_sequence, PackedSequence):
        raise ValueError("Input parameter is not a instance of PackedSequence")
    if th.cuda.is_available():
        packed_sequence = packed_sequence.cuda()
    return packed_sequence


class PITrainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 lr=1e-5,
                 momentum=0.9,
                 weight_decay=0,
                 clip_norm=None,
                 min_lr=0,
                 patience=1,
                 factor=0.5,
                 disturb_std=0.0):
        self.nnet = nnet
        logger.info("Network structure:\n{}".format(self.nnet))
        self.optimizer = create_optimizer(
            optimizer,
            self.nnet.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.nnet.to(device)
        self.checkpoint = checkpoint
        self.num_spks = nnet.num_spks
        self.clip_norm = clip_norm
        self.disturb = disturb_std
        if self.disturb:
            logger.info("Disturb networks with std = {}".format(disturb_std))
        if self.clip_norm:
            logger.info("Clip gradient by 2-norm {}".format(clip_norm))
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)

    def train(self, dataset):
        self.nnet.train()
        logger.info("Training...")
        tot_loss = num_batch = 0
        for input_sizes, nnet_input, source_attr, target_attr in dataset:
            num_batch += 1
            nnet_input = packed_sequence_cuda(nnet_input) if isinstance(
                nnet_input, PackedSequence) else nnet_input.to(device)

            self.optimizer.zero_grad()

            if self.disturb:
                self.nnet.disturb(self.disturb)

            masks = self.nnet(nnet_input)
            cur_loss = self.permutate_loss(masks, input_sizes, source_attr,
                                           target_attr)
            tot_loss += cur_loss.item()

            cur_loss.backward()

            if self.clip_norm:
                th.nn.utils.clip_grad_norm_(self.nnet.parameters(),
                                            self.clip_norm)
            self.optimizer.step()

        return tot_loss / num_batch, num_batch

    def validate(self, dataset):
        self.nnet.eval()
        logger.info("Cross Validate...")
        tot_loss = num_batch = 0
        # do not need to keep gradient
        with th.no_grad():
            for input_sizes, nnet_input, source_attr, target_attr in dataset:
                num_batch += 1
                nnet_input = packed_sequence_cuda(nnet_input) if isinstance(
                    nnet_input, PackedSequence) else nnet_input.to(device)
                masks = self.nnet(nnet_input)
                cur_loss = self.permutate_loss(masks, input_sizes, source_attr,
                                               target_attr)
                tot_loss += cur_loss.item()

        return tot_loss / num_batch, num_batch

    def run(self, train_set, dev_set, num_epoches=20):
        init_loss, _ = self.validate(dev_set)
        logger.info("Epoch {:2d}: dev = {:.4f}".format(0, init_loss))
        th.save(self.nnet.state_dict(),
                os.path.join(self.checkpoint, 'epoch.0.pkl'))
        for epoch in range(1, num_epoches + 1):
            on_train_start = time.time()
            train_loss, train_num_batch = self.train(train_set)
            on_valid_start = time.time()
            valid_loss, valid_num_batch = self.validate(dev_set)
            on_valid_end = time.time()
            # scheduler learning rate
            self.scheduler.step(valid_loss)
            logger.info(
                "Loss(time/mini-batch) - Epoch {:2d}: train = {:.4f}({:.2f}s/{:d}) |"
                " dev = {:.4f}({:.2f}s/{:d})".format(
                    epoch, train_loss, on_valid_start - on_train_start,
                    train_num_batch, valid_loss, on_valid_end - on_valid_start,
                    valid_num_batch))
            save_path = os.path.join(self.checkpoint,
                                     'epoch.{:d}.pkl'.format(epoch))
            th.save(self.nnet.state_dict(), save_path)
        logger.info("Training for {} epoches done!".format(num_epoches))

    def permutate_loss(self, masks, input_sizes, source_attr, target_attr):
        """
        Arguments:
            masks: tensor list on device
            input_sizes: 1D tensor on cpu
            source_attr: python dict: {
                "spectrogram": tensor,
                "phase": tensor, only for psm
            }
            target_attr: python dict: {
                "spectrogram": [tensor...],
                "phase": [tensor...], only for psm
            }
        """
        input_sizes = input_sizes.to(device)
        mixture_spect = source_attr["spectrogram"].to(device)
        targets_spect = [t.to(device) for t in target_attr["spectrogram"]]

        if self.num_spks != len(targets_spect):
            raise ValueError(
                "Number targets do not match known speakers: {} vs {}".format(
                    self.num_spks, len(targets_spect)))

        is_loss_with_psm = "phase" in source_attr
        if is_loss_with_psm:
            mixture_phase = source_attr["phase"].to(device)
            targets_phase = [t.to(device) for t in target_attr["phase"]]

        def loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute):
                # refer_spect = targets_spect[t] * th.cos(
                #     mixture_phase -
                #     targets_phase[t]) if is_loss_with_psm else targets_spect[t]
                # TODO: using non-negative psm(add ReLU)?
                refer_spect = targets_spect[t] * F.relu(
                    th.cos(mixture_phase - targets_phase[t])
                ) if is_loss_with_psm else targets_spect[t]
                # N x T x F => N x 1
                utt_loss = th.sum(
                    th.sum(
                        th.pow(masks[s] * mixture_spect - refer_spect, 2), -1),
                    -1)
                loss_for_permute.append(utt_loss)
            loss_perutt = sum(loss_for_permute) / input_sizes
            return loss_perutt

        num_utts = input_sizes.shape[0]
        # O(N!), could be optimized
        # P x N
        pscore = th.stack(
            [loss(p) for p in permutations(range(self.num_spks))])
        min_perutt, _ = th.min(pscore, dim=0)
        return th.sum(min_perutt) / (self.num_spks * num_utts)