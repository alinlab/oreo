import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import logging
import random
import csv

logging.basicConfig(level=logging.INFO)
import argparse
import matplotlib.pyplot as plt

from linear_models import CoordConvBetaVAE, weight_init
from utils import load_dataset, set_seed_everywhere
from dopamine.discrete_domains.atari_lib import create_atari_environment
import kornia

gfile = tf.io.gfile


def compute_loss(x, x_pred, mu, logvar, kl_tolerance=0):
    recon_loss = (x - x_pred).pow(2).sum([1, 2, 3]).mean(0)
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    kl_loss = torch.clamp(kl_loss, kl_tolerance * mu.shape[1], mu.shape[1]).mean()
    return recon_loss, kl_loss


def train(args):
    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed_everywhere(args.seed)

    observations, actions, _ = load_dataset(
        args.env,
        1,
        args.datapath,
        args.normal,
        args.num_data,
        args.stack,
        args.num_episodes,
    )

    logging.info("Building models..")
    beta_vae = CoordConvBetaVAE(args.z_dim, args.ch_div).to(device)

    if args.lmd > 0:
        env = create_atari_environment(args.env)
        action_dim = env.action_space.n

        actor = nn.Sequential(
            nn.Linear(args.z_dim, args.z_dim),
            nn.ReLU(),
            nn.Linear(args.z_dim, action_dim),
        )
        actor.apply(weight_init)
        actor.to(device)
        if torch.cuda.device_count() > 1:
            actor = nn.DataParallel(actor)

    save_dir = "models_beta_vae"
    resize = kornia.geometry.Resize(64)
    save_dir = save_dir + "_coord_conv_chdiv{}".format(args.ch_div)
    if args.lmd > 0:
        save_dir = save_dir + "_actor_lmd{}".format(args.lmd)
    if args.add_path is not None:
        save_dir = save_dir + "_" + args.add_path

    if args.num_episodes is None:
        save_tag = "{}_s{}_data{}k_con{}_seed{}_zdim{}_beta{}_kltol{}".format(
            args.env,
            args.stack,
            int(args.num_data / 1000),
            1 - int(args.normal),
            args.seed,
            args.z_dim,
            int(args.beta),
            args.kl_tolerance,
        )
    else:
        save_tag = "{}_s{}_epi{}_con{}_seed{}_zdim{}_beta{}_kltol{}".format(
            args.env,
            args.stack,
            int(args.num_episodes),
            1 - int(args.normal),
            args.seed,
            args.z_dim,
            int(args.beta),
            args.kl_tolerance,
        )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## Multi-GPU
    if torch.cuda.device_count() > 1:
        beta_vae = nn.DataParallel(beta_vae)

    if args.lmd > 0:
        beta_vae_optimizer = torch.optim.Adam(
            list(beta_vae.parameters()) + list(actor.parameters()), lr=args.lr
        )
    else:
        beta_vae_optimizer = torch.optim.Adam(beta_vae.parameters(), lr=args.lr)

    n_batch = len(observations) // args.batch_size + 1
    total_idxs = list(range(len(observations)))

    logging.info("Training starts..")
    f = open(os.path.join(save_dir, save_tag + "_beta_vae_train.csv"), "w")
    writer = csv.writer(f)
    if args.lmd > 0:
        writer.writerow(["Epoch", "Recon Error", "KL Loss", "Actor Loss"])
    else:
        writer.writerow(["Epoch", "Recon Error", "KL Loss"])

    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(args.n_epochs)):
        random.shuffle(total_idxs)
        recon_errors = []
        kl_losses = []
        actor_losses = []
        for j in range(n_batch):
            batch_idxs = total_idxs[j * args.batch_size : (j + 1) * args.batch_size]
            xx = torch.as_tensor(
                observations[batch_idxs], device=device, dtype=torch.float32
            )
            xx = xx / 255.0
            xx = resize(xx)

            beta_vae_optimizer.zero_grad()

            z, mu, logvar = beta_vae(xx, mode="encode")
            obs_pred = beta_vae(z, mode="decode")
            recon_loss, kl_loss = compute_loss(
                xx, obs_pred, mu, logvar, args.kl_tolerance
            )

            if args.lmd > 0:
                batch_act = torch.as_tensor(actions[batch_idxs], device=device).long()
                logits = actor(z)
                actor_loss = criterion(logits, batch_act)
                loss = recon_loss + args.beta * kl_loss + args.lmd * actor_loss
                actor_losses.append(actor_loss.mean().detach().cpu().item())
            else:
                loss = recon_loss + args.beta * kl_loss

            loss.backward()

            beta_vae_optimizer.step()

            recon_errors.append(recon_loss.mean().detach().cpu().item())
            kl_losses.append(kl_loss.mean().detach().cpu().item())

        if args.lmd > 0:
            logging.info(
                "Epoch {} | Recon Error: {:.4f} | KL Loss: {:.4f} | Actor Loss: {:.4f}".format(
                    epoch + 1,
                    np.mean(recon_errors),
                    np.mean(kl_losses),
                    np.mean(actor_losses),
                )
            )
            writer.writerow(
                [
                    epoch + 1,
                    np.mean(recon_errors),
                    np.mean(kl_losses),
                    np.mean(actor_losses),
                ]
            )
        else:
            logging.info(
                "Epoch {} | Recon Error: {:.4f} | KL Loss: {:.4f}".format(
                    epoch + 1, np.mean(recon_errors), np.mean(kl_losses)
                )
            )
            writer.writerow([epoch + 1, np.mean(recon_errors), np.mean(kl_losses)])

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                beta_vae.module.state_dict()
                if (torch.cuda.device_count() > 1)
                else beta_vae.state_dict(),
                os.path.join(
                    save_dir, save_tag + "_ep{}_beta_vae.pth".format(epoch + 1)
                ),
            )
            if args.lmd > 0:
                torch.save(
                    actor.module.state_dict()
                    if (torch.cuda.device_count() > 1)
                    else actor.state_dict(),
                    os.path.join(
                        save_dir, save_tag + "_ep{}_actor.pth".format(epoch + 1)
                    ),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Seed & Env
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--env", default="Pong", type=str)
    parser.add_argument("--datapath", default="/data", type=str)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--normal", action="store_true", default=False)
    parser.add_argument("--num_data", default=50000, type=int)
    parser.add_argument("--num_episodes", default=None, type=int)
    parser.add_argument("--stack", default=1, type=int)
    parser.add_argument("--add_path", default=None, type=str)

    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--num_residual_hiddens", default=32, type=int)
    parser.add_argument("--beta", default=4, type=float)
    parser.add_argument("--kl_tolerance", default=0, type=float)
    parser.add_argument("--z_dim", default=50, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--n_epochs", default=1000, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)

    parser.add_argument("--ch_div", default=1, type=int)

    parser.add_argument("--lmd", default=0, type=float)

    args = parser.parse_args()
    assert args.beta > 1.0, "beta should be larger than 1"
    train(args)
