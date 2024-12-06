import numpy as np
import torch
torch.set_num_threads(10)

import cv2
import argparse
import os
import gym
import time
import json

from shutil import copyfile
import utils
from logger import Logger




def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--total_frames', default=1000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='scr', type=str, choices=['scr'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str, choices=['pixel', 'identity'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--note', default='', type=str)

    # n_step
    parser.add_argument('--goal_n_steps', default=None, type=int)
    # distracting_control
    parser.add_argument('--distracting_control', default=False, action='store_true')   
    parser.add_argument('--difficulty', default='easy', type=str, choices=['easy', 'medium', 'hard'])
    
    # save when finish
    parser.add_argument('--save_when_finish', default=True, action='store_true')
    parser.add_argument('--save_finish_step', default=500000, type=int)

    
    args = parser.parse_args()
    print(args)
    return args


def evaluate(env, agent, num_episodes, L, step, device=None,):

    for i in range(num_episodes):


        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)

            obs, reward, done, info = env.step(action)

            episode_reward += reward


        L.log('eval/episode_reward', episode_reward, step)
        if 'success' in info:
            L.log('eval/success', info['success'], step)
        L.dump(step)



def make_agent(obs_shape, action_shape, args, device):
    utils.set_seed_everywhere(args.seed)

    if args.agent == 'scr':
        if args.agent == 'scr':
            import agent.scr
            from agent.scr import AnalogyAgent
            source_code_file_path = agent.scr.__file__
        else:
            raise NotImplementedError
        agent = AnalogyAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef
        )
        agent.source_code_file_path = source_code_file_path
    else:
        raise NotImplementedError

    if args.load_encoder:
        # model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder) 
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    if args.distracting_control:
        import DistractingControlSuite
        env = DistractingControlSuite.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            difficulty=args.difficulty,
            seed=args.seed,
            from_pixels=(args.encoder_type == 'pixel'),
            max_episode_steps=args.total_frames,
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat,
        )
        env.seed(args.seed)
        eval_env = DistractingControlSuite.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            difficulty=args.difficulty,
            seed=args.seed+1,
            from_pixels=(args.encoder_type == 'pixel'),
            max_episode_steps=args.total_frames,
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat,
        )
        eval_env.seed(args.seed)

    else:
        import dmc2gym
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            episode_length=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        env.seed(args.seed)

        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            episode_length=args.total_frames,
            seed=args.seed+1,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    # buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))


    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    if args.agent.startswith('analogy') or args.agent.startswith('ddpg'):
        replay_buffer = utils.ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            gamma=args.discount,
            device=device,
            goal_n_steps=args.goal_n_steps,
            sample_ij=True,
            do_not_swap=True,
        )
    else:
        replay_buffer = utils.ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            gamma=args.discount,
            # gamma=0.95,
            device=device,
        )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    if hasattr(agent, 'source_code_file_path'):
        if agent.source_code_file_path is not None:
            code_dir = os.path.join(args.work_dir, 'code')
            os.makedirs(code_dir, exist_ok=True)
            copyfile(agent.source_code_file_path, os.path.join(code_dir, 
                    os.path.basename(agent.source_code_file_path)))

    episode, episode_reward, done = 0, 0, True
    info = {}
    start_time = time.time()
    learner_step = 0
    for step in range(args.num_train_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                evaluate(eval_env, agent, args.num_eval_episodes, L, step)
                if args.save_model:
                    agent.save(model_dir, step)
            #     if args.save_buffer:
            #         replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)
            if 'success' in info:
                L.log('train/success', info['success'], step)

            obs = env.reset()
            # print(obs.shape)
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)

        # sample action for data collection
        # print(obs.shape)
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, learner_step)
                learner_step += 1

        curr_reward = reward
        next_obs, reward, done, info = env.step(action)
        # cv2.imwrite('fig2/{}.png'.format(step), np.transpose(next_obs[0:3], (1, 2, 0))[..., [2, 1, 0]])

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episodic_exp_terminated = float(done) #or (episode_step + 1 == env._max_episode_steps)
        episode_reward += reward

        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_bool, episode_step, episodic_exp_terminated)
        # np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        obs = next_obs
        episode_step += 1

        if args.save_when_finish and step == args.save_finish_step - 1:
            os.makedirs(model_dir, exist_ok=True)
            agent.save(model_dir, args.save_finish_step)
            


def collect_data(env, agent, num_rollouts, path_length, checkpoint_path):
    rollouts = []
    for i in range(num_rollouts):
        obses = []
        acs = []
        rews = []
        observation = env.reset()
        for j in range(path_length):
            action = agent.sample_action(observation)
            next_observation, reward, done, _ = env.step(action)
            obses.append(observation)
            acs.append(action)
            rews.append(reward)
            observation = next_observation
        obses.append(next_observation)
        rollouts.append((obses, acs, rews))

    from scipy.io import savemat

    savemat(
        os.path.join(checkpoint_path, "dynamics-data.mat"),
        {
            "trajs": np.array([path[0] for path in rollouts]),
            "acs": np.array([path[1] for path in rollouts]),
            "rews": np.array([path[2] for path in rollouts])
        }
    )


if __name__ == '__main__':
    main()
