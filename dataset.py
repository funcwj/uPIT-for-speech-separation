#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import random
import pickle

import numpy as np
import torch as th

from torch.nn.utils.rnn import pack_sequence, pad_sequence

from utils import parse_scps, stft, apply_cmvn, EPSILON, get_logger

logger = get_logger(__name__)


class SpectrogramReader(object):
    """
        Wrapper for short-time fourier transform of dataset
    """

    def __init__(self, wave_scp, **kwargs):
        if not os.path.exists(wave_scp):
            raise FileNotFoundError("Could not find file {}".format(wave_scp))
        self.stft_kwargs = kwargs
        self.wave_dict = parse_scps(wave_scp)
        self.wave_keys = [key for key in self.wave_dict.keys()]
        logger.info(
            "Create SpectrogramReader for {} with {} utterances".format(
                wave_scp, len(self.wave_dict)))

    def __len__(self):
        return len(self.wave_dict)

    def __contains__(self, key):
        return key in self.wave_dict

    # stft
    def _load(self, key):
        return stft(self.wave_dict[key], **self.stft_kwargs)

    # sequential index
    def __iter__(self):
        for key in self.wave_dict:
            yield key, self._load(key)

    # random index
    def __getitem__(self, key):
        if key not in self.wave_dict:
            raise KeyError("Could not find utterance {}".format(key))
        return self._load(key)


class Dataset(object):
    def __init__(self, mixture_reader, targets_reader_list):
        self.mixture_reader = mixture_reader
        self.keys_list = mixture_reader.wave_keys
        self.targets_reader_list = targets_reader_list
        self.num_spks = len(targets_reader_list)

    def __len__(self):
        return len(self.keys_list)

    def _has_target(self, key):
        for targets_reader in self.targets_reader_list:
            if key not in targets_reader:
                return False
        return True

    def _index_by_key(self, key):
        """
            Return a tuple like (matrix, [matrix, ...])
        """
        if key not in self.mixture_reader or not self._has_target(key):
            raise KeyError("Missing targets or mixture")
        target_list = [reader[key] for reader in self.targets_reader_list]
        return (self.mixture_reader[key], target_list)

    def _index_by_num(self, num):
        """
            Return a tuple like (matrix, [matrix, ...])
        """
        if num >= len(self.keys_list):
            raise IndexError("Index out of dataset, {} vs {}".format(
                num, len(self.keys_list)))
        key = self.keys_list[num]
        return self._index_by_key(key)

    def _index_by_list(self, list_idx):
        """
            Returns a list of tuple like [
                (matrix, [matrix, ...]),
                (matrix, [matrix, ...]),
                ...
            ]
        """
        if max(list_idx) >= len(self.keys_list):
            raise IndexError("Index list contains index out of dataset")
        return [self._index_by_num(index) for index in list_idx]

    def __getitem__(self, index):
        """
            Implement to support multi-type index: by key, number or list
        """
        if type(index) == int:
            return self._index_by_num(index)
        elif type(index) == str:
            return self._index_by_key(index)
        elif type(index) == list:
            return self._index_by_list(index)
        else:
            raise KeyError("Unsupported index type(int/str/list)")


class BatchSampler(object):
    def __init__(self,
                 sampler_size,
                 batch_size=16,
                 shuffle=True,
                 drop_last=False):
        if batch_size <= 0:
            raise ValueError(
                "Illegal batch_size(= {}) detected".format(batch_size))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler_index = list(range(sampler_size))
        self.sampler_size = sampler_size
        if shuffle:
            random.shuffle(self.sampler_index)

    def __len__(self):
        return self.sampler_size

    def __iter__(self):
        base = 0
        step = self.batch_size
        while True:
            if base + step > self.sampler_size:
                break
            yield (self.sampler_index[base:base + step]
                   if step != 1 else self.sampler_index[base])
            base += step
        if not self.drop_last and base < self.sampler_size:
            yield self.sampler_index[base:]


class DataLoader(object):
    """
        Multi utterance loader for permutation invariant training
        Support both AM and PSM
    """

    def __init__(self,
                 dataset,
                 shuffle=True,
                 batch_size=16,
                 drop_last=False,
                 apply_log=True,
                 mvn_dict=None):
        if batch_size == 1:
            raise ValueError("Now do not support perutt training")
        self.dataset = dataset
        self.mvn_dict = mvn_dict
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_spks = dataset.num_spks
        if mvn_dict:
            logger.info("Using cmvn dictionary from {}".format(mvn_dict))
            with open(mvn_dict, "rb") as f:
                self.mvn_dict = pickle.load(f)
        # use log-spectrogram instead, cause we do not apply-log in SpectrogramReader
        self.apply_log = apply_log

    def __len__(self):
        remain = len(self.dataset) % self.batch_size
        if self.drop_last or not remain:
            return len(self.dataset) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size + 1

    def _transform(self, mixture_specs, targets_specs_list):
        """
        Transform original spectrogram
            If mixture_specs is a complex object, it means PAM will be used for training
            It can be configured in .yaml, egs: apply_abs=false to produce complex results
            If mixture_specs is real, we will using AM(ratio mask)

        Arguments:
            mixture_specs: non-log complex/real spectrogram
            targets_specs_list: list of non-log complex/real spectrogram for each target speakers
        Returns:
            python dictionary with four attributes:
            num_frames: length of current utterance
            feature: input feature for networks, egs: log spectrogram + cmvn
            source_attr: a dictionary with at most 2 keys: spectrogram and phase(for PSM), each contains a tensor
            target_attr: same keys like source_attr, each keys correspond to a tensor list
        """
        # NOTE: mixture_specs may be complex or real
        input_spectra = np.abs(mixture_specs) if np.iscomplexobj(
            mixture_specs) else mixture_specs
        # apply_log and cmvn, for nnet input
        if self.apply_log:
            input_spectra = np.log(np.maximum(input_spectra, EPSILON))
        if self.mvn_dict:
            input_spectra = apply_cmvn(input_spectra, self.mvn_dict)

        # using dict to pack infomation needed in loss
        source_attr = {}
        target_attr = {}

        if np.iscomplexobj(mixture_specs):
            source_attr["spectrogram"] = th.tensor(
                np.abs(mixture_specs), dtype=th.float32)
            target_attr["spectrogram"] = [
                th.tensor(np.abs(t), dtype=th.float32)
                for t in targets_specs_list
            ]
            source_attr["phase"] = th.tensor(
                np.angle(mixture_specs), dtype=th.float32)
            target_attr["phase"] = [
                th.tensor(np.angle(t), dtype=th.float32)
                for t in targets_specs_list
            ]
        else:
            source_attr["spectrogram"] = th.tensor(
                mixture_specs, dtype=th.float32)
            target_attr["spectrogram"] = [
                th.tensor(t, dtype=th.float32) for t in targets_specs_list
            ]

        return {
            "num_frames": mixture_specs.shape[0],
            "feature": th.tensor(input_spectra, dtype=th.float32),
            "source_attr": source_attr,
            "target_attr": target_attr
        }

    def _process(self, index):
        """
        Transform utterance index into a minbatch

        Arguments:
            index: a list type

        Returns:
            input_sizes: a tensor correspond to utterance length
            input_feats: packed sequence to feed networks
            source_attr/target_attr: dictionary contains spectrogram/phase needed in loss computation
        """
        if type(index) is not list:
            raise ValueError("Unsupported index type({})".format(type(index)))

        def prepare_target(dict_list, index, key):
            return pad_sequence(
                [d["target_attr"][key][index] for d in dict_list],
                batch_first=True)

        # sorted by utterance
        dict_list = sorted(
            [self._transform(s, t) for s, t in self.dataset[index]],
            key=lambda x: x["num_frames"],
            reverse=True)
        # pack tensor for network input
        input_feats = pack_sequence([d["feature"] for d in dict_list])
        input_sizes = th.tensor(
            [d["num_frames"] for d in dict_list], dtype=th.float32)

        source_attr = {}
        target_attr = {}

        source_attr["spectrogram"] = pad_sequence(
            [d["source_attr"]["spectrogram"] for d in dict_list],
            batch_first=True)
        target_attr["spectrogram"] = [
            prepare_target(dict_list, s, "spectrogram")
            for s in range(self.num_spks)
        ]

        if "phase" in dict_list[0]:
            source_attr["phase"] = pad_sequence(
                [d["source_attr"]["phase"] for d in dict_list],
                batch_first=True)
            target_attr["phase"] = [
                prepare_target(dict_list, s, "phase")
                for s in range(self.num_spks)
            ]

        return input_sizes, input_feats, source_attr, target_attr

    def __iter__(self):
        sampler = BatchSampler(
            len(self.dataset),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last)
        num_utts = 0
        log_period = 2000 // self.batch_size
        for e, index in enumerate(sampler):
            num_utts += (len(index) if type(index) is list else 1)
            if not (e + 1) % log_period:
                logger.info("Processed {} batches, {} utterances".format(
                    e + 1, num_utts))
            yield self._process(index)
        logger.info("Processed {} utterances in total".format(num_utts))
