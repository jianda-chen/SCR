import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import kornia

import utils
# from sac_analoy import  Actor, Critic, LOG_FREQ
from sac_pixel import  Actor, Critic, LOG_FREQ
from transition_model import make_transition_model

from torchqmet import IQE, QuasimetricBase

EPSILON = 1e-9

def _sqrt(x, tol=EPSILON):
    tol = torch.ones_like(x)*tol
    return torch.sqrt(torch.maximum(x, tol))

def cosine_distance(x, y):
    numerator = torch.sum(x * y, dim=-1, keepdim=True)
    denominator = torch.sqrt(
        torch.sum(x.pow(2.), dim=-1, keepdim=True)) * torch.sqrt(torch.sum(y.pow(2.), dim=-1, keepdim=True))
    cos_similarity = numerator / (denominator + EPSILON)

    return torch.atan2(_sqrt(1. - cos_similarity.pow(2.)), cos_similarity)

def value_rescale(value, eps=1e-3):
        return value.sign()*((value.abs() + 1.).sqrt() - 1.) + eps * value

def inverse_value_rescale(value, eps=1e-3):
    temp = ((1 + 4 * eps * (value.abs() + 1. + eps)).sqrt() - 1.) / (2. * eps)
    return value.sign() * (temp.square() - 1.)

class AnalogyAgent(object):
    """Bisimulation metric algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        transition_model_type,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        bisim_coef=0.5
    ):
        print(__file__)
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef
        self.encoder_feature_dim = encoder_feature_dim
        # self.mico_structural_distance="mico_angular"
        # self.mico_structural_distance="x^2+x^2-xy"
        self.mico_structural_distance="-xy_norm" # *
        # self.mico_structural_distance="l1_smooth"
        # self.mico_structural_distance="simsr"


        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)
        
        self.dim_per_component = 64
        self.iqe = IQE(
            encoder_feature_dim, 
            dim_per_component=self.dim_per_component, 
            reduction='maxmean').to(device)
        
        self.dual_mlp = nn.Sequential(
            nn.Linear(encoder_feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        ).to(device)

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()) + list(self.iqe.parameters()) + list(self.dual_mlp.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for encoder
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        # critic_loss = F.smooth_l1_loss(current_Q1,
        #                          target_Q) + F.smooth_l1_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        L.log('train_alpha/entropy',  (self.alpha.detach() * log_pi).mean().item(), step)
       
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, obs, h, action, reward, next_obs, next_h, interval_reward_sum, L, step):

        # Sample random states across episodes at random
        batch_size = obs.size(0) // 2
        perm0 = torch.randperm(batch_size)

        perm = torch.cat((perm0, perm0))
        h2 = h[perm]

        h_single = h
        h_single_i, h_single_j = h_single[:batch_size], h_single[batch_size:]
        h_dual = self.dual_mlp(torch.cat((h_single_i, h_single_j), dim=-1))
              
        
        d_ij = self.iqe(h_single_i, h_single_j).unsqueeze(-1)

        loss_m = 0.
        d_ij_rescale = False
        if d_ij_rescale:
            ## value rescale
            rescale_interval_reward_sum = value_rescale(interval_reward_sum)
            loss_lhs_no_reduce = (F.mse_loss(d_ij, rescale_interval_reward_sum, reduction='none') * (d_ij - rescale_interval_reward_sum < 0.0).detach())
        else:
            loss_lhs_no_reduce = (F.mse_loss(d_ij, interval_reward_sum, reduction='none') * (d_ij - interval_reward_sum < 0.0).detach())


        perm0 = torch.randperm(batch_size)
        # (i, j) - (i', j') <= d(i,i') + d(j',j')
        # (i, j)  <= d(i,i') + d(j',j') + (i', j')
        
        if d_ij_rescale:
            ## value rescale
            with torch.no_grad():
                rhs = value_rescale(self.metric_func(h_single_i, h_single_i[perm0]) + self.metric_func(h_single_j, h_single_j[perm0]) + inverse_value_rescale(d_ij[perm0]).relu())
        else:
            with torch.no_grad():
                rhs = self.metric_func(h_single_i, h_single_i[perm0]) + self.metric_func(h_single_j, h_single_j[perm0]) + d_ij[perm0].relu()
        rhs_error = F.mse_loss(d_ij, rhs, reduction='none') * (d_ij > rhs).detach()

        loss_m += (rhs_error).mean()
        loss_m += 1e-3 * (loss_lhs_no_reduce).mean() 

        L.log('train_ae/encoder_d_ij', d_ij.mean().item(), step)
        if d_ij_rescale:
            L.log('train_ae/encoder_lhs', rescale_interval_reward_sum.mean().item(), step)
            L.log('train_ae/encoder_rhs_lhs', rhs.mean().item() - rescale_interval_reward_sum.mean().item(), step)
        else:
            L.log('train_ae/encoder_lhs', interval_reward_sum.mean().item(), step)
            L.log('train_ae/encoder_rhs_lhs', rhs.mean().item() - interval_reward_sum.mean().item(), step)
        L.log('train_ae/encoder_rhs', rhs.mean().item(), step)

        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h_single, action], dim=1))
            reward2 = reward[perm]
        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        z_dist = self.metric_func(h_single, h_single[perm])


        r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        transition_dist = self.metric_func(pred_next_latent_mu1, pred_next_latent_mu2)

        bisimilarity = r_dist + self.discount * transition_dist
        loss_phi = (z_dist - bisimilarity).pow(2).mean()

        L.log('train_ae/encoder_loss_phi', loss_phi, step)

        perm0 = torch.randperm(batch_size)
        with torch.no_grad():
            r_i = reward[:batch_size]
            r_dist_i = F.smooth_l1_loss(r_i, r_i[perm0], reduction='none')

            # # predicted next latent at i+1
            pred_next_latent_mu1_i = pred_next_latent_mu1[:batch_size]
            h_dual_pred_next = self.dual_mlp(torch.cat((pred_next_latent_mu1_i, h_single_j), dim=-1)) #(pred_ip1, j)
            # bisimilarity_i = value_rescale(r_dist_i + self.discount * inverse_value_rescale(self.metric_func(h_dual_pred_next, h_dual_pred_next[perm0])))
            bisimilarity_i = r_dist_i + self.discount * self.metric_func(h_dual_pred_next, h_dual_pred_next[perm0])


        loss_psi = F.mse_loss(self.metric_func(h_dual, h_dual[perm0]), # |(x_i. x_j), (y_i, y_j)|
                                      bisimilarity_i.detach()) # |r_i^x - r_i^y| + gamma * |(x_ip1. x_j), (y_ip1, y_j)|
        


        

        loss = 1.0 * (loss_phi + loss_psi + loss_m)
        L.log('train_ae/encoder_loss_phi', loss_phi, step)
        L.log('train_ae/encoder_loss_psi', loss_psi, step)
        L.log('train_ae/encoder_loss_m', loss_m, step)
        L.log('train_ae/encoder_all_loss', loss, step)
        return loss

    def update_transition_reward_model(self, obs, h, action, next_obs, next_h, reward, L, step):
        h = h[..., :h.shape[-1]]
        next_h = next_h[..., :next_h.shape[-1]]
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', loss, step)

        pred_next_latent = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        pred_next_reward = self.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        return total_loss

    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done, interval_reward_sum = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)


        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        h = self.critic.encoder(obs)
        next_h = self.critic.encoder(next_obs)
        transition_reward_loss = self.update_transition_reward_model(obs, h, action, next_obs, next_h, reward, L, step)
        encoder_loss = 0.1 * self.update_encoder(obs, h, action, reward, next_obs, next_h, interval_reward_sum, L, step)
        total_loss = self.bisim_coef * encoder_loss + transition_reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

            
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        )

    def metric_func(self, x, y, mico_structural_distance=None):
        if mico_structural_distance is None:
            mico_structural_distance = self.mico_structural_distance
        if mico_structural_distance == 'l2':
            dist = F.pairwise_distance(x, y, p=2, keepdim=True)
        elif mico_structural_distance == 'l1_smooth':
            dist = F.smooth_l1_loss(x, y, reduction='none')
            dist = dist.mean(dim=-1, keepdim=True)
        elif mico_structural_distance == 'mico_angular':
            beta = 1e-6 
            base_distances = cosine_distance(x, y)
            norm_average = (x.pow(2.).sum(dim=-1, keepdim=True)
                + y.pow(2.).sum(dim=-1, keepdim=True))
            dist = norm_average + beta * base_distances
        elif mico_structural_distance == 'simsr':
            dist = 1. - (x * y).sum(dim=-1, keepdim=True)
        elif mico_structural_distance == 'x^2+x^2-xy':
            k = 0.1 # 0 < k < 2
            base_distances = (x * y).sum(dim=-1, keepdim=True)
            norm_average = (x.pow(2.).sum(dim=-1, keepdim=True) 
                + y.pow(2.).sum(dim=-1, keepdim=True))
            dist = norm_average - k * base_distances
        elif mico_structural_distance == '-xy_norm':
            k = 1.
            base_distances = (x * y).sum(dim=-1, keepdim=True)
            norm_average = (x.pow(2.).sum(dim=-1, keepdim=True) 
                + y.pow(2.).sum(dim=-1, keepdim=True))
            dist = norm_average - k * base_distances
            dist = dist.sqrt()
        
        else:

            raise NotImplementedError
        return dist