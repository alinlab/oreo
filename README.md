# OREO: Object-Aware Regularization for Addressing Causal Confusion in Imitation Learning (NeurIPS 2021)

## Video demo
We here provide a video demo from confounded Enduro environment (see Figure 8 of the main draft).
We also visualize the spatial attention map from a convolutional encoder trained with BC (medium) and OREO (right). 

![Enduro_total_demo_cropped](https://user-images.githubusercontent.com/33256298/120595374-38554300-c47d-11eb-97e5-afff3c5d83b9.gif)

## Installation

OREO requires CUDA 10.1 to run. 

Install the dependencies:
```sh
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install dopamine_rl sklearn tqdm kornia dropblock atari-py==0.2.6 gsutil
```

Download DQN Replay dataset for expert demonstrations on Atari environments:
```sh
mkdir DATAPATH
cp download.sh DATAPATH
cd DATAPATH
sh download.sh
```


## Pre-training
We here provide beta-VAE (for CCIL) and VQ-VAE (for CRLR and OREO) pretraining scripts. 
For other datasets, change the --env option.

### beta-VAE
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python atari_beta_vae.py --env=KungFuMaster --datapath DATAPATH --num_episodes 20 --seed 1 --ch_div 4 --lmd 10
```
### VQ-VAE
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python atari_vqvae.py --env=KungFuMaster --datapath DATAPATH --num_episodes 20 --seed 1
```

## Training BC policy
We here provide training scripts for baselines and OREO. 
For other datasets, change the --env, --beta_vae_path, and --vqvae_path options.

### Behavioral cloning
```sh
CUDA_VISIBLE_DEVICES=0 python atari_cnn_actor.py --env=KungFuMaster --datapath DATAPATH --seed 1 --eval_interval 1000 --num_episodes 20 --num_eval_episodes 100
```
### Dropout
```sh
CUDA_VISIBLE_DEVICES=0 python atari_cnn_actor.py --env=KungFuMaster --datapath DATAPATH --seed 1 --eval_interval 1000 --original_dropout --prob 0.5 --num_episodes 20 --num_eval_episodes 100
```
### DropBlock
```sh
CUDA_VISIBLE_DEVICES=0 python atari_cnn_actor.py --env=KungFuMaster --datapath DATAPATH --seed 1 --eval_interval 1000 --dropblock --prob 0.3 --num_episodes 20 --num_eval_episodes 100
```
### Cutout
```sh
CUDA_VISIBLE_DEVICES=0 python atari_cnn_actor.py --env=KungFuMaster --datapath DATAPATH --seed 1 --eval_interval 1000 --input_cutout --num_episodes 20 --num_eval_episodes 100
```
### RandomShift
```sh
CUDA_VISIBLE_DEVICES=0 python atari_cnn_actor.py --env=KungFuMaster --datapath DATAPATH --seed 1 --eval_interval 1000 --random_shift --num_episodes 20 --num_eval_episodes 100
```
### CCIL (w/o interaction)
```sh
CUDA_VISIBLE_DEVICES=0 python atari_beta_vae_actor.py --env=KungFuMaster --datapath DATAPATH --num_episodes 20 --num_eval_episodes 100 --seed 1 --eval_interval 1000 --prob 0.5 --ch_div 4 --beta_vae_path models_beta_vae_coord_conv_chdiv4_actor_lmd10.0/KungFuMaster_s1_epi20_con1_seed1_zdim50_beta4_kltol0_ep1000_beta_vae.pth
```
### CRLR
```sh
CUDA_VISIBLE_DEVICES=0 python atari_cnn_actor_crlr.py --fixed_size 15000 --num_sub_iters 10 --eval_interval 10 --save_interval 10 --n_epochs 10 --env=KungFuMaster --datapath DATAPATH --num_episodes 20 --num_eval_episodes 100 --seed 1 --vqvae_path models_vqvae/KungFuMaster_s1_epi20_con1_seed1_ne512_c0.25_ep1000_vqvae.pth
```
## OREO
```sh
CUDA_VISIBLE_DEVICES=0 python atari_vqvae_oreo.py --env=KungFuMaster --datapath DATAPATH --num_mask 5 --num_episodes 20 --num_eval_episodes 100 --seed 1 --eval_interval 1000 --prob 0.5 --vqvae_path models_vqvae/KungFuMaster_s1_epi20_con1_seed1_ne512_c0.25_ep1000_vqvae.pth
```
