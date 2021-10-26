import gzip
import os
import logging
import random
from tqdm import tqdm
from collections import deque
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import gym

logging.basicConfig(level=logging.INFO)
import pickle

from PIL import Image, ImageFont, ImageDraw
from sklearn.linear_model import Ridge
from torch.distributions import Bernoulli
import kornia

gfile = tf.io.gfile


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(env, seed, datapath, normal, num_data, stack, num_episodes=None):
    try:
        if num_episodes is not None:
            path = os.path.join(
                datapath,
                env,
                str(seed),
                "replay_logs",
                "saved_episodes_{}_normal{}.pkl".format(int(num_episodes), int(normal)),
            )
        else:
            path = os.path.join(
                datapath,
                env,
                str(seed),
                "replay_logs",
                "saved_dataset_{}_normal{}.pkl".format(int(num_data), int(normal)),
            )
        with open(path, "rb") as f:
            observations, actions, data_variance = pickle.load(f)
    except Exception as e:
        print(e)
        path = os.path.join(datapath, env, str(seed), "replay_logs")
        ckpts = gfile.listdir(path)
        observation_lists = [os.path.join(path, p) for p in ckpts if "observation" in p]
        observation_lists = sorted(
            observation_lists, key=lambda s: int(s.split(".")[-2])
        )
        action_lists = [os.path.join(path, p) for p in ckpts if "action" in p]
        action_lists = sorted(action_lists, key=lambda s: int(s.split(".")[-2]))
        terminal_lists = [os.path.join(path, p) for p in ckpts if "terminal" in p]
        terminal_lists = sorted(terminal_lists, key=lambda s: int(s.split(".")[-2]))

        logging.info("Loading observations..")
        o_ckpt = observation_lists[-1]
        with tf.io.gfile.GFile(o_ckpt, "rb") as f:
            with gzip.GzipFile(fileobj=f) as infile:
                obs_chunk = np.load(infile, allow_pickle=False)

        logging.info("Loading actions..")
        a_ckpt = action_lists[-1]
        with tf.io.gfile.GFile(a_ckpt, "rb") as f:
            with gzip.GzipFile(fileobj=f) as infile:
                act_chunk = np.load(infile, allow_pickle=False)
        logging.info("Loading terminals..")
        t_ckpt = terminal_lists[-1]
        with tf.io.gfile.GFile(t_ckpt, "rb") as f:
            with gzip.GzipFile(fileobj=f) as infile:
                terminal_chunk = np.load(infile, allow_pickle=False)

        if num_episodes is not None:
            cut_idxs = np.where(terminal_chunk != 0)[0] + 1
            # list of episodes
            observations = np.split(obs_chunk, cut_idxs)[1:-1]
            actions = np.split(act_chunk, cut_idxs)[1:-1]
            terminals = np.split(terminal_chunk, cut_idxs)[1:-1]

            total_episodes = len(observations)
            num_episodes = min(int(num_episodes), total_episodes)
            logging.info("Number of episodes: {}".format(num_episodes))
            observations = observations[: int(num_episodes)]
            actions = actions[: int(num_episodes)]
            terminals = terminals[: int(num_episodes)]

            observations = np.concatenate(observations, 0)
            actions = np.concatenate(actions, 0)
            terminals = np.concatenate(terminals, 0)
            logging.info("Number of frames: {}".format(len(observations)))

            data_variance = np.var(
                observations[: min(len(observations), 100000)] / 255.0
            )
        else:
            observations = obs_chunk[: int(num_data)]
            actions = act_chunk[: int(num_data)]
            terminals = terminal_chunk[: int(num_data)]

            data_variance = np.var(observations[: min(int(num_data), 100000)] / 255.0)

        logging.info("Stacking dataset..")
        stacked_obs = []
        stacked_actions = []
        previous_actions = []
        i = stack
        terminal_cnt = 0
        while True:
            if terminals[i] == 0:
                stacked_obs.append(observations[i - stack + 1 : i + 1])
                stacked_actions.append(actions[i])
                previous_actions.append(actions[i - 1])
                i += 1
            else:
                terminal_cnt += 1
                i += stack
            if i >= len(observations):
                break
        observations = np.array(stacked_obs)
        actions = np.array(stacked_actions)

        logging.info("Number of terminals: {}".format(terminal_cnt))

        if not normal:
            confounded_observations = np.empty(
                shape=(observations.shape[0], *observations.shape[1:]),
                dtype=observations.dtype,
            )
            logging.info("Building dataset with previous actions to the images..")
            for i in tqdm(range(observations.shape[0])):
                if stack != 1:
                    img = Image.fromarray(np.transpose(observations[i], (1, 2, 0)))
                else:
                    img = Image.fromarray(observations[i][0])
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("arial.ttf", size=16)
                draw.text(
                    (11, 55),
                    "{}".format(previous_actions[i]),
                    fill=(255,) * stack,
                    font=font,
                )
                if stack != 1:
                    confounded_observations[i] = np.transpose(
                        np.asarray(img), (2, 0, 1)
                    )
                else:
                    confounded_observations[i] = np.asarray(img)[None, ...]

            observations = confounded_observations

        if num_episodes is not None:
            path = os.path.join(
                datapath,
                env,
                str(seed),
                "replay_logs",
                "saved_episodes_{}_normal{}.pkl".format(int(num_episodes), int(normal)),
            )
            with open(path, "wb") as f:
                pickle.dump([observations, actions, data_variance], f, protocol=4)
        else:
            path = os.path.join(
                datapath,
                env,
                str(seed),
                "replay_logs",
                "saved_dataset_{}_normal{}.pkl".format(int(num_data), int(normal)),
            )
            with open(path, "wb") as f:
                pickle.dump([observations, actions, data_variance], f, protocol=4)

    logging.info("Done!")
    assert observations.shape[0] == actions.shape[0], (
        observations.shape,
        actions.shape,
    )
    return observations, actions, data_variance


class StackedObs:
    def __init__(self, stack, confounded):
        self._stack = stack
        self._confounded = confounded
        self._deque = deque(maxlen=stack)
        self._font = ImageFont.truetype("arial.ttf", size=16)

    def reset(self, obs):
        self._deque.clear()
        for _ in range(self._stack):
            self._deque.append(obs)
        prev_action = 0
        return self._get_stacked_obs(prev_action)

    def step(self, obs, prev_action):
        self._deque.append(obs)
        return self._get_stacked_obs(prev_action)

    def _get_stacked_obs(self, prev_action):
        if self._confounded:
            stacked_obs = []
            for c in range(self._stack):
                img = Image.fromarray(self._deque[c][..., 0])
                draw = ImageDraw.Draw(img)
                draw.text(
                    (11, 55), "{}".format(prev_action), fill=255, font=self._font,
                )
                obs = np.asarray(img)[..., None]
                stacked_obs.append(obs)
            stacked_obs = np.concatenate(stacked_obs, axis=2)
        else:
            stacked_obs = np.concatenate(self._deque, axis=2)
        stacked_obs = np.transpose(stacked_obs, (2, 0, 1))
        return stacked_obs


def sample(weights, temperature):
    return (
        Bernoulli(logits=torch.from_numpy(weights) / temperature)
        .sample()
        .long()
        .numpy()
    )


def linear_regression(masks, rewards, alpha=1.0):
    model = Ridge(alpha).fit(masks, rewards)
    return model.coef_, model.intercept_


def evaluate(
    env,
    pre_actor,
    actor,
    model,
    mode,
    device,
    args,
    topk_index=None,
    mask=None,
    num_eval_episodes=None,
    quantizer=None,
):
    model.eval()
    actor.eval()
    stacked_obs_factory = StackedObs(args.stack, not args.normal_eval)
    average_episode_reward = 0
    if num_eval_episodes is None:
        num_eval_episodes = args.num_eval_episodes

    if hasattr(args, "coord_conv"):
        resize = kornia.geometry.Resize(64)

    for episode in range(num_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            if step == 0:
                stacked_obs = stacked_obs_factory.reset(obs)

            with torch.no_grad():
                stacked_obs = (
                    torch.as_tensor(
                        stacked_obs, device=device, dtype=torch.float32
                    ).unsqueeze(0)
                    / 255.0
                )
                if hasattr(args, "coord_conv"):
                    if args.coord_conv:
                        stacked_obs = resize(stacked_obs)

                if mode in ["cnn", "beta_vae"]:
                    features = model(stacked_obs)
                else:
                    raise NotImplementedError(mode)

                if mode == "cnn":
                    if quantizer is not None:
                        features = quantizer(features)[0]
                    features = pre_actor(features)
                    action = actor(features).argmax(1)[0].cpu().item()
                elif mode == "beta_vae":
                    features = pre_actor(torch.flatten(features, start_dim=1))
                    features, _ = features.chunk(2, dim=-1)  # mu
                    features = torch.cat([features, torch.ones_like(features)], dim=1)
                    action = actor(features).argmax(1)[0].cpu().item()
                else:
                    raise NotImplementedError(mode)

            obs, reward, done, info = env.step(action)
            prev_action = action
            stacked_obs = stacked_obs_factory.step(obs, prev_action)
            episode_reward += reward
            step += 1
            if step == 27000:
                done = True

        average_episode_reward += episode_reward
    average_episode_reward /= num_eval_episodes
    model.train()
    actor.train()
    return average_episode_reward


def evaluate_crlr(
    env, actor, model, encoder, quantizer, device, args, num_eval_episodes=None,
):
    model.eval()
    actor.eval()
    encoder.eval()
    quantizer.eval()
    stacked_obs_factory = StackedObs(args.stack, not args.normal_eval)
    average_episode_reward = 0
    if num_eval_episodes is None:
        num_eval_episodes = args.num_eval_episodes
    for episode in tqdm(range(num_eval_episodes)):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            if step == 0:
                stacked_obs = stacked_obs_factory.reset(obs)

            with torch.no_grad():
                stacked_obs = (
                    torch.as_tensor(
                        stacked_obs, device=device, dtype=torch.float32
                    ).unsqueeze(0)
                    / 255.0
                )

                z = encoder(stacked_obs)
                z, *_, encoding_indices, _ = quantizer(z)
                # features = model(stacked_obs)

                ## one-hot encoding
                encoding_indices_flatten = encoding_indices.view(-1)  # (Bx64)
                encoding_indices_onehot = torch.zeros(
                    (len(encoding_indices_flatten), args.num_embeddings),
                    device=encoding_indices_flatten.device,
                )
                encoding_indices_onehot.scatter_(
                    1, encoding_indices_flatten.unsqueeze(1), 1
                )
                encoding_indices_onehot = encoding_indices_onehot.view(
                    1, -1, args.num_embeddings
                )  # B x 64 x 512

                logits = actor(torch.flatten(encoding_indices_onehot, start_dim=1))
                action = logits.argmax(1)[0].cpu().item()

            obs, reward, done, info = env.step(action)
            prev_action = action
            stacked_obs = stacked_obs_factory.step(obs, prev_action)
            episode_reward += reward
            step += 1
            if step == 27000:
                done = True

        average_episode_reward += episode_reward
    average_episode_reward /= num_eval_episodes
    model.train()
    actor.train()
    return average_episode_reward


def categorical_confounder_balancing_loss(x, w, num_classes, x_onehot=None):
    N, P = x.shape

    # one-hot encoding
    if x_onehot is None:
        is_treat = torch.zeros((N * P, num_classes), device=x.device)
        is_treat.scatter_(1, x.reshape(-1).unsqueeze(1), 1)
        is_treat = is_treat.view(N, P, num_classes)
        is_treat = is_treat.permute(2, 0, 1)  # NPC -> CNP
    else:
        is_treat = x_onehot.permute(2, 0, 1)

    w = w.unsqueeze(0).repeat(num_classes, 1)  # N -> CN

    ## CPN x (CN1 * CNP) * CPP = CPP
    target_set = torch.bmm(
        is_treat.permute(0, 2, 1), F.normalize(w.unsqueeze(2) * is_treat, p=1, dim=1)
    ) * ~torch.eye(P, dtype=bool, device=x.device).unsqueeze(0).repeat(
        num_classes, 1, 1
    )
    target_set = target_set.permute(1, 2, 0)  # CPP -> PPC
    target_set = target_set.reshape(P, -1)  # P(PC)
    loss = torch.sum(torch.var(target_set, dim=0))

    return loss
