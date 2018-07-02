#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os

import numpy as np

from dataset import SpectrogramReader
from utils import istft


def compute_mask(mixture, targets_list, mask_type):
    """
    Arguments:
        mixture: STFT of mixture signal(complex result) 
        targets_list: python list of target signal's STFT results(complex result)
        mask_type: ["irm", "ibm", "iam", "psm"]
    """
    if mask_type == 'ibm':
        max_index = np.argmax(
            np.stack([np.abs(mat) for mat in targets_list]), 0)
        return [max_index == s for s in range(len(targets_list))]

    if mask_type == "irm":
        denominator = sum([np.abs(mat) for mat in targets_list])
    else:
        denominator = np.abs(mixture)
    if mask_type != "psm":
        masks = [np.abs(mat) / denominator for mat in targets_list]
    else:
        mixture_phase = np.angle(mixture)
        masks = [
            np.abs(mat) * np.cos(mixture_phase - np.angle(mat)) / denominator
            for mat in targets_list
        ]
    return masks


def run(args):
    # return complex result
    reader_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": True
    }
    print("Using Mask: {}".format(args.mask.upper()))
    mixture_reader = SpectrogramReader(
        args.mix_scp, **reader_kwargs, return_samps=True)
    targets_reader = [
        SpectrogramReader(scp, **reader_kwargs) for scp in args.ref_scp
    ]
    num_utts = 0
    for key, packed in mixture_reader:
        samps, mixture = packed
        norm = np.linalg.norm(samps, np.inf)
        skip = False
        for reader in targets_reader:
            if key not in reader:
                print("Skip utterance {}, missing targets".format(key))
                skip = True
                break
        if skip:
            continue
        num_utts += 1
        if not num_utts % 1000:
            print("Processed {} utterance...".format(num_utts))
        targets_list = [reader[key] for reader in targets_reader]
        spk_masks = compute_mask(mixture, targets_list, args.mask)
        for index, mask in enumerate(spk_masks):
            istft(
                os.path.join(args.dump_dir, '{}.spk{}.wav'.format(
                    key, index + 1)),
                mixture * mask,
                **reader_kwargs,
                norm=norm,
                fs=8000,
                nsamps=samps.size)
    print("Processed {} utterance!".format(num_utts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Command to do oracle speech separation, using specified mask(IAM|IBM|IRM|PSM)"
    )
    parser.add_argument(
        "mix_scp",
        type=str,
        help="Location of mixture wave scripts in kaldi format")
    parser.add_argument(
        "ref_scp",
        nargs="+",
        help="List of reference speaker wave scripts in kaldi format")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="cache",
        dest="dump_dir",
        help="Location to dump seperated speakers")
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=128,
        dest="frame_shift",
        help="Number of samples shifted when spliting frames")
    parser.add_argument(
        "--frame-length",
        type=int,
        default=256,
        dest="frame_length",
        help="Frame length in number of samples")
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        dest="window",
        help="Type of window function, see scipy.signal.get_window")
    parser.add_argument(
        "--mask",
        type=str,
        dest="mask",
        default="irm",
        choices=["iam", 'irm', 'ibm', 'psm'],
        help="Type of mask to use for speech separation")
    args = parser.parse_args()
    run(args)