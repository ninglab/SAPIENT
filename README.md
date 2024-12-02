# SAPIENT
Implementation of the paper "SAPIENT: Mastering Multi-turn Conversational Recommendation with Strategic Planning and Monte Carlo Tree Search"

##Introduction
Conversational Recommender Systems (CRS) proactively engage users in interactive dialogues to elicit user preferences and provide personalized recommendations. Existing methods train Reinforcement Learning (RL)-based agent with greedy action selection or sampling strategy, and may suffer from suboptimal conversational planning. To address this, we present a novel Monte Carlo Tree Search (MCTS)-based CRS framework SAPIENT. SAPIENT consists of a conversational agent (S-agent) and a conversational planner (S-planner). S-planner builds a conversational search tree with MCTS based on the initial actions proposed by S-agent to find conversation plans. The best conversation plans from S-planner are used to guide the training of S-agent, creating a self-training loop where S-agent can iteratively improve its capability for conversational planning. Furthermore, we propose an efficient variant SAPIENT-e for trade-off between training efficiency and performance. Extensive experiments on four benchmark datasets validate the effectiveness of our approach, showing that SAPIENT outperforms the state-of-the-art baselines.

## Training and Evaluation
`python RL_model.py --data_name <LAST_FM_STAR, YELP_STAR, BOOK, MOVIE>`

## SAPIENT-e
The implementation of SAPIENT-e is in the folder /efficiency.

## Dataset Preparation

The preprocessed datasets are in the folder /data

You may use the following steps to preprocess your own dataset:

1. Put the user-item interaction data into the folder /data/<data_name>, you may refer to the details in [SCPR](https://github.com/farrecall/SCPR).
2. Preprocess the dataset: `python graph_init.py --data_name <data_name>`
3. Use TransE in [[OpenKE](https://github.com/thunlp/OpenKE)] to pretrain the graph embeddings. Then put the pretrained embeddings under "/tmp/<data_name>/embeds/".

## Requirements
dgl==0.9.1.post1

easydict==1.13

ipdb==0.13.13

numpy==1.26.4

torch==2.3.0+cu121

tqdm==4.66.2
