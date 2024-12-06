#!/bin/bash

DOMAIN=cheetah
TASK=run
ACTION_REPEAT=4
# DOMAIN=walker
# TASK=walk
# ACTION_REPEAT=2
# DOMAIN=cartpole
# TASK=swingup
# ACTION_REPEAT=8
# DOMAIN=finger
# TASK=spin
# ACTION_REPEAT=2
# DOMAIN=ball_in_cup
# TASK=catch
# ACTION_REPEAT=4
# DOMAIN=reacher
# TASK=easy
# ACTION_REPEAT=4
# DOMAIN=cartpole
# TASK=swingup_sparse
# ACTION_REPEAT=8
# DOMAIN=hopper
# TASK=stand
# ACTION_REPEAT=4


DIFFICULTY='easy'
# DIFFICULTY='medium'
# DIFFICULTY='hard'


SAVEDIR=../log_github

NOW=$(date +"%Y-%m-%d-%H-%M-%S")
echo ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW}


python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --distracting_control \
    --difficulty ${DIFFICULTY} \
    --agent 'scr' \
    --init_steps 1000 \
    --num_train_steps 510000 \
    --encoder_type pixel \
    --transition_model_type 'probabilistic' \
    --encoder_feature_dim 512 \
    --action_repeat ${ACTION_REPEAT} \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --hidden_dim 1024 \
    --total_frames 1000 \
    --num_layers 4 \
    --num_filters 32 \
    --batch_size 128 \
    --encoder_lr 1e-4 \
    --decoder_lr 1e-4 \
    --actor_lr 1e-4 \
    --critic_lr 1e-4 \
    --alpha_beta 0.5 \
    --alpha_lr 1e-4 \
    --init_temperature 0.1 \
    --num_eval_episodes 5 \
    --discount 0.99 \
    --goal_n_steps 100 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW} \
    --seed 201 "$@"
