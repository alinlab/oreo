from pprint import pprint

import os
import random
import csv
import argparse
from time import perf_counter
from collections import deque

from PIL import Image, ImageFont, ImageDraw
import numpy as np
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from dopamine.discrete_domains.atari_lib import create_atari_environment
from linear_models import Encoder, CoordConvEncoder
import kornia
from utils import set_seed_everywhere


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


class SoftQAlgo:
    def __init__(
        self, num_dims, reward_fn, its, temperature=1.0, device=None, evals_per_it=1,
    ):
        self.num_dims = num_dims
        self.reward_fn = reward_fn
        self.its = its
        self.device = device
        self.temperature = lambda t: temperature
        self.evals_per_it = evals_per_it

    def run(self, args, writer):
        t = self.temperature(0)
        weights = np.zeros(self.num_dims)

        trace = []
        masks = []
        rewards = []
        steps = []

        mode = (np.sign(weights).astype(np.int64) + 1) // 2
        score = np.mean(
            [self.reward_fn(mode)[0] for _ in range(args.num_eval_episodes)]
        )
        writer.writerow([args.env, args.seed, 0, 0, score])

        for it in range(self.its):
            start = perf_counter()
            mask = sample(weights, t)
            reward = []
            step = []
            for _ in range(self.evals_per_it):
                r, s = self.reward_fn(mask)
                reward.append(r)
                step.append(s)
            reward, step = np.mean(reward), np.sum(step)

            masks.append(mask)
            rewards.append(reward)
            steps.append(step)

            weights, _ = linear_regression(masks, rewards, alpha=1.0)

            mode = (np.sign(weights).astype(np.int64) + 1) // 2
            trace.append(
                {
                    "it": it,
                    "reward": reward,
                    "mask": mask,
                    "weights": weights,
                    "mode": mode,
                    "time": perf_counter() - start,
                    "past_mean_reward": np.mean(rewards),
                }
            )
            pprint(trace[-1])

            if (it + 1) % args.eval_interval == 0:
                score = np.mean(
                    [self.reward_fn(mode)[0] for _ in range(args.num_eval_episodes)]
                )
                print()
                total_steps = np.sum(steps)
                print(f"Reward at iter {it+1}, interaction {total_steps}: {score}")
                print()
                writer.writerow([args.env, args.seed, it + 1, total_steps, score])

        return trace


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


def evaluate(env, pre_actor, actor, model, mask, device, args, num_eval_episodes):
    model.eval()
    actor.eval()
    stacked_obs_factory = StackedObs(args.stack, not args.normal_eval)
    average_episode_reward = 0
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    human_scores = {
        "Amidar": 1675.8,
        "Asterix": 8503.3,
        "CrazyClimber": 35410.5,
        "DemonAttack": 3401.3,
        "Enduro": 309.6,
        "Freeway": 29.6,
        "Gopher": 2321.0,
        "Jamesbond": 406.7,
        "Kangaroo": 3035.0,
        "KungFuMaster": 22736.2,
        "Pong": 9.3,
        "PrivateEye": 69571.3,
        "Seaquest": 20181.8,
        "Alien": 6875.4,
        "Assault": 1496.4,
        "BankHeist": 734.4,
        "BattleZone": 37800.0,
        "Boxing": 4.3,
        "Breakout": 31.8,
        "ChopperCommand": 9881.8,
        "Frostbite": 4334.7,
        "Hero": 25762.5,
        "Krull": 2394.6,
        "MsPacman": 15693.4,
        "Qbert": 13455.0,
        "RoadRunner": 7845.0,
        "UpNDown": 9082.0,
    }
    random_scores = {
        "Amidar": 5.8,
        "Asterix": 210.0,
        "CrazyClimber": 10780.5,
        "DemonAttack": 152.1,
        "Enduro": 0.0,
        "Freeway": 0.0,
        "Gopher": 257.6,
        "Jamesbond": 29.0,
        "Kangaroo": 52.0,
        "KungFuMaster": 258.5,
        "Pong": -20.7,
        "PrivateEye": 24.9,
        "Seaquest": 68.4,
        "Alien": 227.8,
        "Assault": 222.4,
        "BankHeist": 14.2,
        "BattleZone": 2360.0,
        "Boxing": 0.1,
        "Breakout": 1.7,
        "ChopperCommand": 811.0,
        "Frostbite": 65.2,
        "Hero": 1027.0,
        "Krull": 1598.0,
        "MsPacman": 307.3,
        "Qbert": 163.9,
        "RoadRunner": 11.5,
        "UpNDown": 533.4,
    }

    resize = kornia.geometry.Resize(64)
    total_step = 0
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

                stacked_obs = resize(stacked_obs)
                features = model(stacked_obs)

                features = pre_actor(torch.flatten(features, start_dim=1))
                features, _ = features.chunk(2, dim=-1)  # mu
                # causal graph
                features = torch.cat(
                    [features * mask, mask.repeat(features.shape[0], 1)], dim=1
                )
                action = actor(features).argmax(1)[0].cpu().item()

            obs, reward, done, info = env.step(action)
            prev_action = action
            stacked_obs = stacked_obs_factory.step(obs, prev_action)
            episode_reward += reward
            step += 1
            if step == 27000:
                done = True
        total_step += step

        average_episode_reward += episode_reward
    average_episode_reward /= num_eval_episodes
    model.train()
    actor.train()
    normalized_reward = (average_episode_reward - random_scores[args.env]) / np.abs(
        human_scores[args.env] - random_scores[args.env]
    )
    return normalized_reward, total_step


def intervention_policy_execution(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed_everywhere(args.seed)

    device = torch.device("cuda")
    env = create_atari_environment(args.env)
    action_dim = env.action_space.n

    actor = nn.Sequential(
        nn.Linear(args.z_dim * 2, args.z_dim),
        nn.ReLU(),
        nn.Linear(args.z_dim, action_dim),
    ).to(device)
    encoder = CoordConvEncoder(1, args.z_dim * 2, args.ch_div).to(device)

    if args.env in [
        "Amidar",
        "Asterix",
        "CrazyClimber",
        "DemonAttack",
        "Enduro",
        "Freeway",
        "Gopher",
        "Jamesbond",
        "Kangaroo",
        "KungFuMaster",
        "Pong",
        "PrivateEye",
        "Seaquest",
    ]:
        num_episodes = 20
    elif args.env in [
        "Alien",
        "Assault",
        "BankHeist",
        "BattleZone",
        "Boxing",
        "Breakout",
        "ChopperCommand",
        "Frostbite",
        "Hero",
        "Krull",
        "MsPacman",
        "Qbert",
        "RoadRunner",
        "UpNDown",
    ]:
        num_episodes = 50
    else:
        raise ValueError("not a target game")

    encoder_path = os.path.join(
        args.save_path,
        "{}_s1_epi{}_con{}_seed{}_ne512_prob0.5_ep1000_encoder.pth".format(
            args.env, num_episodes, 1 - int(args.normal_eval), args.seed
        ),
    )
    actor_path = os.path.join(
        args.save_path,
        "{}_s1_epi{}_con{}_seed{}_ne512_prob0.5_ep1000_actor.pth".format(
            args.env, num_episodes, 1 - int(args.normal_eval), args.seed
        ),
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))

    ## Multi-GPU
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        actor = nn.DataParallel(actor)

    def run_step(mask):
        score, steps = evaluate(
            env,
            nn.Identity(),
            actor.module if torch.cuda.device_count() > 1 else actor,
            encoder.module if torch.cuda.device_count() > 1 else encoder,
            mask,
            device,
            args,
            1,
        )
        return score, steps

    save_dir = "models_beta_vae_actor_coord_conv_ccil_normalized"
    save_tag = "{}_s{}_epi{}_con{}_seed{}_ne{}_temp{}".format(
        args.env,
        args.stack,
        int(num_episodes),
        1 - int(args.normal_eval),
        args.seed,
        args.num_embeddings,
        int(args.temperature),
    )

    if args.add_path is not None:
        save_dir = save_dir + "_" + args.add_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    f_te = open(os.path.join(save_dir, save_tag + "_cnn_eval.csv"), "w")
    writer_te = csv.writer(f_te)
    writer_te.writerow(["Game", "Seed", "Iters", "Interactions", "Score"])

    trace = SoftQAlgo(
        args.z_dim, run_step, args.num_its, temperature=args.temperature
    ).run(args, writer_te)

    best_mask = trace[-1]["mode"]
    print(f"Final mask {best_mask.tolist()}")

    score, _ = evaluate(
        env,
        nn.Identity(),
        actor.module if torch.cuda.device_count() > 1 else actor,
        encoder.module if torch.cuda.device_count() > 1 else encoder,
        best_mask,
        device,
        args,
        args.num_eval_episodes,
    )

    print(f"Final reward {score}")
    writer_te.writerow([args.env, args.seed, args.num_its, "final", score])

    f_te.close()

    torch.save(
        torch.from_numpy(best_mask),
        os.path.join(save_dir, save_tag + "_best_mask.pth"),
    )

    print(f"Final reward {score}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_its", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=10)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--env", default="Pong", type=str)
    parser.add_argument("--datapath", default="/data", type=str)
    parser.add_argument("--stack", default=1, type=int)
    parser.add_argument("--normal_eval", action="store_true", default=False)

    # Save & Evaluation
    parser.add_argument("--num_eval_episodes", default=20, type=int)
    parser.add_argument("--eval_interval", default=20, type=int)
    parser.add_argument("--add_path", default=None, type=str)

    # Encoder & Hyperparams
    parser.add_argument("--num_embeddings", default=512, type=int)
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--num_residual_hiddens", default=32, type=int)

    # Model load
    parser.add_argument("--save_path", default=None, type=str)
    # For MLP
    parser.add_argument("--z_dim", default=50, type=int)
    parser.add_argument("--ch_div", default=1, type=int)

    intervention_policy_execution(parser.parse_args())


if __name__ == "__main__":
    main()
