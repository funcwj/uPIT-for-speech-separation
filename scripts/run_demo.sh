#!/usr/bin/env bash

mix_scp=./data/2spk/test/mix.scp 
mdl_dir=./tune/2spk_pit_a

set -eu

[ -d ./cache ] && rm -rf cache

mkdir cache

shuf $mix_scp | head -n30 > egs.scp

./separate.py --dump-dir cache $mdl_dir/train.yaml $mdl_dir/epoch.100.pkl egs.scp

rm -f egs.scp
