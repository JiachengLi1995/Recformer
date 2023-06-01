# Recformer

This repository contains the code of Recformer, a model learns natural language representations for sequential recommendation.

Our KDD 2023 paper [Text Is All You Need: Learning Language Representations for Sequential Recommendation](https://arxiv.org/abs/2305.13731).

## Quick Links

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Pretraining](#pretraining)
- [Pretrained Model](#pretrained-model)
- [Finetuning](#finetuning)
- [Contact](#contact)
- [Citation](#citation)

## Overview

In this paper, we propose to model user preferences and item features as language representations that can be generalized to new items and datasets. To this end, we present a novel framework, named Recformer, which effectively learns language representations for sequential recommendation. Specifically, we propose to formulate an item as a "sentence" (word sequence) by flattening item key-value attributes described by text so that an item sequence for a user becomes a sequence of sentences. For recommendation, Recformer is trained to understand the "sentence" sequence and retrieve the next "sentence". To encode item sequences, we design a bi-directional Transformer similar to the model Longformer but with different embedding layers for sequential recommendation. For effective representation learning, we propose novel pretraining and finetuning methods which combine language understanding and recommendation tasks. Therefore, Recformer can effectively recommend the next item based on language representations.

## Dependencies

We train and test the model using the following main dependencies:
- Python 3.10.10
- PyTorch 2.0.0
- PyTorch Lightning 2.0.0
- Transformers 4.28.0
- Deepspeed 0.9.0

## Pretraining
### Dataset
We use 8 categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) for pretraining:

Training:
- `Automotive`
- `Cell Phones and Accessories`
- `Clothing, Shoes and Jewelry`
- `Electronics`
- `Grocery and Gourmet Food`
- `Home and Kitchen`
- `Movies and TV`

Validation:
- `CDs and Vinyl`

You can process these data using our provided scripts `pretrain_data/meta_data_process.py` and `pretrain_data/interaction_data_process.py`. You need to set meta data path `META_ROOT` and interaction data path `SEQ_ROOT` in the two files. Then run the following commands:
```bash
cd pretrain_data
python meta_data_process.py
python interaction_data_process.py
```
Or, you can download our processed data from [here](https://drive.google.com/file/d/11wTD3jMoP_Fb5SlHfKr28NIMCnG_jOpy/view?usp=sharing).

### Training

Our pretraining code is based on the framework [Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/). Our backbone model is `allenai/longformer-base-4096` but we have different `token type embedding` and `item position embedding`.

First, we need to adjust pretrained Longformer checkpoint to our model. You can run the following command:
```bash
python save_longformer_ckpt.py
```
This code will automatically download `allenai/longformer-base-4096` from Huggingface then adjust and save it to `longformer_ckpt/longformer-base-4096.bin`.

Then, you can pretrain your own model with our default settings by running the following command:
```bash
bash lightning_run.sh
```
If you use the training strategy `deepspeed_stage_2` (default setting in our script), you need to first convert zero checkpoint to lightning checkpoint by running `zero_to_fp32.py` (automatically generated to checkpoint folder from pytorch-lightning):
```bash
python zero_to_fp32.py . pytorch_model.bin
```
Finally, we convert the lightning checkpoint to pytorch checkpoint (they have different model parameter names) by running `convert_pretrain_ckpt.py`:
```bash
python convert_pretrain_ckpt.py
```
You need to set four paths in the file: 
- `LIGHTNING_CKPT_PATH`, pretrained lightning checkpoint path.
- `LONGFORMER_CKPT_PATH`, Longformer checkpoint (from `save_longformer_ckpt.py`) path.
- `OUTPUT_CKPT_PATH`, output path of Recformer checkpoint (for class `RecformerModel` in `recformer/models.py`).
- `OUTPUT_CONFIG_PATH`, output path of Recformer for Sequential Recommendation checkpoint (for class `RecformerForSeqRec` in `recformer/models.py`). 

## Pretrained Model

We provide pretrained checkpoints for `RecformerModel` and `RecformerForSeqRec` used in our paper (`allenai/longformer-base-4096` as backbone).
|              Model              |
|:-------------------------------|
|[RecformerModel](https://drive.google.com/file/d/1aWsPLLgBaO51mPqzZrNdPmlBkMEZ-naR/view?usp=sharing)|
|[RecformerForSeqRec](https://drive.google.com/file/d/1BEboY3NxAUOBe6YwYZ_RsQ4BR6IIbl0-/view?usp=sharing)|

You can load the pretrained model by running the following code:
```python
import torch
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec

config = RecformerConfig.from_pretrained('allenai/longformer-base-4096')
config.max_attr_num = 3  # max number of attributes for each item
config.max_attr_length = 32 # max number of tokens for each attribute
config.max_item_embeddings = 51 # max number of items in a sequence +1 for cls token
config.attention_window = [64] * 12 # attention window for each layer

model = RecformerModel(config)
model.load_state_dict(torch.load('recformer_ckpt.bin'))

model = RecformerForSeqRec(config)
model.load_state_dict(torch.load('recformer_seqrec_ckpt.bin'), strict=False)
# strict=False because RecformerForSeqRec doesn't have lm_head
```

## Finetuning
### Dataset
We use 6 categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) to evaluate our model:

- `Industrial and Scientific`
- `Musical Instruments`
- `Arts, Crafts and Sewing`
- `Office Products`
- `Video Games`
- `Pet Supplies`

You can process these data using our provided scripts `finetune_data/process.py`. You need to set meta data path `--meta_file_path`, interaction data path `--file_path` and output path `--output_path` to run the following commands:
```bash
cd finetune_data
python process.py --meta_file_path META_PATH --file_path SEQ_PATH --output_path OUTPUT_FOLDER
```

We also provide all processed data used in our paper [here](https://drive.google.com/file/d/123AHjsvZFTeT_Mhfb81eMHvnE8fbsFi3/view?usp=sharing).

### Training
We train `RecformerForSeqRec` with two-stage finetuning in our paper to conduct the sequential recommendation with Recformer. A sample script is provided for finetuning:
```bash
bash finetune.sh
```
Our code will train and evaluate the model for the sequential recommendation task and return all metrics used in our paper.

## Contact

If you have any questions related to the code or the paper, feel free to email Jiacheng (`j9li@eng.ucsd.edu`).

## Citation

Please cite our paper if you use UCEpic in your work:

```bibtex
@inproceedings{li23text,
  title = "Text is all you need: Learning language representations for sequential recommendation",
  author = "Jiacheng Li and Ming Wang and Jin Li and Jinmiao Fu and Xin Shen and Jingbo Shang and Julian McAuley",
  year = "2023",
  booktitle = "KDD"
}
```