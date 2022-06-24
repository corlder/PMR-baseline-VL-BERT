# VL-BERT for PMR
This is a baseline model of PMR based on VL-BERT, which is adapted from [this repo](https://github.com/jackroos/VL-BERT), many thanks to the authors for open source resource.
## Requirements
Follow the [instruction](https://github.com/jackroos/VL-BERT#prepare) to install the requirements, but please note that `Apex` is necessary to training with our code.
## Data
Please arrange the data in `data/` as follows. `images/` is the directory of `images.zip` after unzipping.
```
data/
├── pmr
│   ├── images/
│   ├── test-ori-without-label.jsonl
│   ├── train-adv.jsonl
│   ├── train-ori.jsonl
│   ├── val-adv.jsonl
│   └── val-ori.jsonl
└── PREPARE_DATA.md
```
## Pre-trained weights
We conducted experiments with `vl-bert-base-e2e.model` and `vl-bert-large-e2e.model`. You can download all necessary pre-trained models and find more details [here](https://github.com/jackroos/VL-BERT/blob/master/model/pretrained_model/PREPARE_PRETRAINED_MODELS.md), and files are supposed to be arranged like this.
```
model/
└── pretrained_model
    ├── bert-base-uncased
    │   ├── bert_config.json
    │   ├── pytorch_model.bin
    │   └── vocab.txt
    ├── bert-large-uncased
    │   ├── bert_config.json
    │   ├── pytorch_model.bin
    │   └── vocab.txt
    ├── PREPARE_PRETRAINED_MODELS.md
    ├── resnet101-pt-vgbua-0000.model
    └── vl-bert-large-e2e.model
```
## Hyperparameters
You can check the configuration and modify of hyperparameters in yaml files under `./cfgs/pmr/`. `pmr_(ori|adv)_data_train.yaml` corresponds with training with original/adversarial data, which are modified from [large_q2a_4x16G_fp32.yaml](https://github.com/jackroos/VL-BERT/tree/master/cfgs/vcr)
## Training
Here is a case of performing single node distributed traininng (we used 4 x A40 in our experiment), and please modify the num_gpus both in this command and config file (e.g. `pmr_ori_data_train.yaml`).
```
./scripts/dist_run_single.sh <num_gpus> pmr/train_end2end.py cfgs/pmr/pmr_ori_data_train.yaml <dir_to_store_checkpoint>
```
## Inference
```
python pmr/test.py \
	--a-cfg ./cfgs/pmr/pmr_ori_data_train.yaml \
	--a-ckpt <checkpoint_of_q2a> \
	--gpus <indexes_of_gpus_to_use> \
	--result-path <dir_to_save_result> --result-name <result_file_name>
```
