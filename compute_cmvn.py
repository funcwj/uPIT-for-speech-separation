#!/usr/bin/env python
# coding=utf-8

# wujian@2018

import argparse
import pickle
import tqdm
import numpy as np

from dataset import SpectrogramReader
from utils import nfft

def run(args):
    reader_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": False,
        "apply_abs": True,
        "apply_log": args.apply_log,
        "apply_pow": args.apply_pow
    }
    num_bins = nfft(args.frame_length) // 2 + 1
    reader = SpectrogramReader(args.wave_scp, **reader_kwargs)
    mean = np.zeros(num_bins)
    std = np.zeros(num_bins)
    num_frames = 0
    # D(X) = E(X^2) - E(X)^2
    for _, spectrogram in tqdm.tqdm(reader):
        num_frames += spectrogram.shape[0]
        mean += np.sum(spectrogram, 0)
        std += np.sum(spectrogram**2, 0)
    mean = mean / num_frames
    std = np.sqrt(std / num_frames - mean**2)
    with open(args.cmvn_dst, "wb") as f:
        cmvn_dict = {"mean": mean, "std": std}
        pickle.dump(cmvn_dict, f)
    print("Totally processed {} frames".format(num_frames))
    print("Global mean: {}".format(mean))
    print("Global std: {}".format(std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to compute global cmvn stats")
    parser.add_argument(
        "wave_scp", type=str, help="Location of mixture wave scripts")
    parser.add_argument(
        "cmvn_dst", type=str, help="Location to dump cmvn stats")
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
        "--apply-log",
        action="store_true",
        default=False,
        dest="apply_log",
        help="If true, using log spectrogram")
    parser.add_argument(
        "--apply-pow",
        action="store_true",
        default=False,
        dest="apply_pow",
        help="If true, using power spectrum instead of magnitude spectrum")
    args = parser.parse_args()
    run(args)