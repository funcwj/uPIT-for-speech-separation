## Speech Separation with uPIT

Speech separation with utterance-level (Permutation Invariant Training)PIT

### requirements

see [requirements.txt](requirements.txt)

### Usage

1. training
    ```shell
    ./run_pit.py --config $conf --num-epoches 50 > $checkpoint/train.log 2>&1 &
    ```

2. inference
    ```
    ./separate.py --dump-dir cache $mdl_dir/train.yaml $mdl_dir/epoch.40.pkl egs.scp
    ```

### Experiments

| Configure | Mask | Epoch |  FM   |  FF  |  MM  | FF/MM | AVG  |
| :-------: | :--: | :---: | :---: | :--: | :--: | :---: | :--: |
| [config-1](conf/1.config.yaml) |  AM  |  40   | 10.17 | 6.38 | 7.05 |  6.72  | 8.54 |

### Reference

* Kolbæk M, Yu D, Tan Z H, et al. Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks[J]. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 2017, 25(10): 1901-1913.