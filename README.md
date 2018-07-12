## Speech Separation with uPIT

Speech separation with utterance-level PIT(Permutation Invariant Training)

### Requirements

see [requirements.txt](requirements.txt)

### Usage

1. Generate dataset using [create-speaker-mixtures.zip](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip)

2. Prepare cmvn, .scp and configure experiments in .yaml files

3. Training:
    ```shell
    ./run_pit.py --config $conf --num-epoches 100 > $checkpoint/train.log 2>&1 &
    ```

4. Inference:
    ```
    ./separate.py --dump-dir cache $mdl_dir/train.yaml $mdl_dir/epoch.40.pkl egs.scp
    ```

### Experiments

| Configure | Mask | Epoch |  FM   |  FF  |  MM  | FF/MM | AVG  |
| :-------: | :--: | :---: | :---: | :--: | :--: | :---: | :--: |
| [config-1](conf/1.config.yaml) |  AM-ReLU    |  75   | 10.41 |  6.73 |  7.35 | 7.19  | 8.82  |
| [config-2](conf/2.config.yaml) |  AM-sigmoid |  50   | 9.95  |  5.99 |  6.72 | 6.35  | 8.26  |
| [config-3](conf/3.config.yaml) |  PSM-ReLU   |  73   | 10.29 |  6.54 |  7.28 | 7.09  | 8.71  |
| [config-4](conf/4.config.yaml) |  PSM-ReLU   |  80   | 10.37 |  6.59 |  7.29 | 7.10  | 8.76  |
| [config-5](conf/5.config.yaml) |  PSM-ReLU   |  62   | 10.58 |  7.00 |  7.55 | 7.40  | 9.01  |
| [config-6](conf/6.config.yaml) |  PSM-ReLU   |  62   | 10.47 |  7.44 |  7.78 | 7.69  | 9.10  |
| [config-7](conf/7.config.yaml) |  PSM-ReLU   |  61   | 10.43 |  7.17 |  7.41 | 7.34  | 8.91  |
|             -                  |  IAM-oracle |   -   | 12.49 | 12.73 | 11.58 | 11.88 | 12.19 |
|             -                  |  IBM-oracle |   -   | 12.94 | 13.20 | 12.04 | 12.35 | 12.65 |
|             -                  |  IRM-oracle |   -   | 12.86 | 13.14 | 11.96 | 12.27 | 12.57 |
|             -                  |  PSM-oracle |   -   | 15.79 | 16.03 | 14.90 | 15.20 | 15.50 |


### Reference

* Kolbæk M, Yu D, Tan Z H, et al. Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks[J]. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 2017, 25(10): 1901-1913.