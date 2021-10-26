import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gzip
import os
import logging
import csv
import random
import copy

logging.basicConfig(level=logging.INFO)
import pickle
import argparse
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw

from linear_models import Encoder, VectorQuantizer, weight_init
from utils import (
    load_dataset,
    evaluate,
    set_seed_everywhere,
)
from dopamine.discrete_domains.atari_lib import create_atari_environment

gfile = tf.io.gfile


def train(args):
    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed_everywhere(args.seed)

    ## fixed dataset
    observations, actions, data_variance = load_dataset(
        args.env,
        1,
        args.datapath,
        args.normal,
        args.num_data,
        args.stack,
        args.num_episodes,
    )

    ## Stage 1
    logging.info("Building models..")
    logging.info("Start stage 1...")

    env = create_atari_environment(args.env)
    action_dim = env.action_space.n

    n_batch = len(observations) // args.batch_size + 1
    total_idxs = list(range(len(observations)))

    logging.info("Training starts..")

    save_dir = "models_vqvae_cnn_actor"
    if args.num_episodes is None:
        save_tag = "{}_s{}_data{}k_con{}_seed{}_ne{}".format(
            args.env,
            args.stack,
            int(args.num_data / 1000),
            1 - int(args.normal),
            args.seed,
            args.num_embeddings,
        )
    else:
        save_tag = "{}_s{}_epi{}_con{}_seed{}_ne{}".format(
            args.env,
            args.stack,
            int(args.num_episodes),
            1 - int(args.normal),
            args.seed,
            args.num_embeddings,
        )

    assert args.vqvae_path is not None
    save_dir = save_dir + "_oreo"
    save_tag = save_tag + "_prob{}".format(args.prob)

    if args.num_mask > 1:
        save_dir = save_dir + "_train_mask{}".format(args.num_mask)

    if args.add_path is not None:
        save_dir = save_dir + "_" + args.add_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    encoder = Encoder(
        args.stack,
        args.embedding_dim,
        args.num_hiddens,
        args.num_residual_layers,
        args.num_residual_hiddens,
    ).to(device)
    quantizer = VectorQuantizer(args.embedding_dim, args.num_embeddings, 0.25).to(
        device
    )
    pre_actor = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(8 * 8 * args.embedding_dim, args.z_dim)
    )
    actor = nn.Sequential(
        nn.Linear(args.z_dim, args.z_dim), nn.ReLU(), nn.Linear(args.z_dim, action_dim),
    )
    pre_actor.apply(weight_init)
    pre_actor.to(device)
    actor.apply(weight_init)
    actor.to(device)

    for p in quantizer.parameters():
        p.requires_grad = False
    vqvae_dict = torch.load(args.vqvae_path, map_location="cpu")
    encoder.load_state_dict(
        {k[9:]: v for k, v in vqvae_dict.items() if "_encoder" in k}
    )
    quantizer.load_state_dict(
        {k[11:]: v for k, v in vqvae_dict.items() if "_quantizer" in k}
    )

    criterion = nn.CrossEntropyLoss()
    scores = []
    logging.info("Training starts..")
    f_tr = open(os.path.join(save_dir, save_tag + "_cnn_train.csv"), "w")
    writer_tr = csv.writer(f_tr)
    writer_tr.writerow(["Epoch", "Loss", "Accuracy"])

    f_te = open(os.path.join(save_dir, save_tag + "_cnn_eval.csv"), "w")
    writer_te = csv.writer(f_te)
    writer_te.writerow(["Epoch", "Loss", "Accuracy", "Score"])

    encoder.eval()
    quantizer.eval()
    total_encoding_indices = []
    with torch.no_grad():
        for j in range(n_batch):
            batch_idxs = total_idxs[j * args.batch_size : (j + 1) * args.batch_size]
            xx = torch.as_tensor(
                observations[batch_idxs], device=device, dtype=torch.float32
            )
            xx = xx / 255.0

            z = encoder(xx)
            z, *_, encoding_indices, _ = quantizer(z)
            total_encoding_indices.append(encoding_indices.cpu())
    total_encoding_indices = torch.cat(total_encoding_indices, dim=0)

    del quantizer

    actor_optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(pre_actor.parameters())
        + list(actor.parameters()),
        lr=args.lr,
    )

    ## Multi-GPU
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        pre_actor = nn.DataParallel(pre_actor)
        actor = nn.DataParallel(actor)

    for epoch in tqdm(range(args.n_epochs)):
        encoder.train()
        actor.train()
        random.shuffle(total_idxs)
        actor_losses = []
        accuracies = []
        for j in range(n_batch):
            batch_idxs = total_idxs[j * args.batch_size : (j + 1) * args.batch_size]
            xx = torch.as_tensor(
                observations[batch_idxs], device=device, dtype=torch.float32
            )
            xx = xx / 255.0
            batch_act = torch.as_tensor(actions[batch_idxs], device=device).long()

            actor_optimizer.zero_grad()

            z = encoder(xx)
            with torch.no_grad():
                encoding_indices = total_encoding_indices[batch_idxs].to(
                    device
                )  # B x 64
                prob = torch.ones(xx.shape[0] * args.num_mask, args.num_embeddings) * (
                    1 - args.prob
                )
                code_mask = torch.bernoulli(prob).to(device)  # B x 512

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
                    xx.shape[0], -1, args.num_embeddings
                )  # B x 64 x 512

                mask = (
                    code_mask.unsqueeze(1)
                    * torch.cat(
                        [encoding_indices_onehot for m in range(args.num_mask)], dim=0
                    )
                ).sum(2)
                mask = mask.reshape(-1, 8, 8)

            z = torch.cat([z for m in range(args.num_mask)], dim=0) * mask.unsqueeze(1)
            z = z / (1.0 - args.prob)
            z = pre_actor(z)

            logits = actor(z)
            actor_loss = criterion(
                logits, torch.cat([batch_act for m in range(args.num_mask)], dim=0)
            )

            actor_loss.backward()

            actor_optimizer.step()

            accuracy = (batch_act == logits[: xx.shape[0]].argmax(1)).float().mean()

            actor_losses.append(actor_loss.mean().detach().cpu().item())
            accuracies.append(accuracy.mean().detach().cpu().item())

        logging.info(
            "(Train) Epoch {} | Actor Loss: {:.4f} | Accuracy: {:.2f}".format(
                epoch + 1, np.mean(actor_losses), np.mean(accuracies),
            )
        )
        writer_tr.writerow(
            [epoch + 1, np.mean(actor_losses), np.mean(accuracies),]
        )

        if (epoch + 1) % args.eval_interval == 0:
            actor.eval()
            encoder.eval()
            score = evaluate(
                env,
                pre_actor.module if torch.cuda.device_count() > 1 else pre_actor,
                actor.module if torch.cuda.device_count() > 1 else actor,
                encoder.module if torch.cuda.device_count() > 1 else encoder,
                "cnn",
                device,
                args,
            )
            logging.info("(Eval) Epoch {} | Score: {:.2f}".format(epoch + 1, score,))
            scores.append(score)
            actor.train()
            encoder.train()
            writer_te.writerow(
                [epoch + 1, np.mean(actor_losses), np.mean(accuracies), score]
            )

    f_tr.close()
    f_te.close()
    torch.save(
        encoder.module.state_dict()
        if torch.cuda.device_count() > 1
        else encoder.state_dict(),
        os.path.join(save_dir, save_tag + "_ep{}_encoder.pth".format(epoch + 1)),
    )
    torch.save(
        actor.module.state_dict()
        if torch.cuda.device_count() > 1
        else actor.state_dict(),
        os.path.join(save_dir, save_tag + "_ep{}_actor.pth".format(epoch + 1),),
    )
    torch.save(
        pre_actor.module.state_dict()
        if torch.cuda.device_count() > 1
        else pre_actor.state_dict(),
        os.path.join(save_dir, save_tag + "_ep{}_pre_actor.pth".format(epoch + 1),),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Seed & Env
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--env", default="Pong", type=str)
    parser.add_argument("--datapath", default="/data", type=str)
    parser.add_argument("--num_data", default=50000, type=int)
    parser.add_argument("--stack", default=1, type=int)
    parser.add_argument("--normal", action="store_true", default=False)
    parser.add_argument("--normal_eval", action="store_true", default=False)

    # Save & Evaluation
    parser.add_argument("--save_interval", default=20, type=int)
    parser.add_argument("--eval_interval", default=20, type=int)
    parser.add_argument("--num_episodes", default=None, type=int)
    parser.add_argument("--num_eval_episodes", default=20, type=int)
    parser.add_argument("--n_epochs", default=1000, type=int)
    parser.add_argument("--add_path", default=None, type=str)

    # Encoder & Hyperparams
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_embeddings", default=512, type=int)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--num_residual_hiddens", default=32, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)

    # Model load
    parser.add_argument("--vqvae_path", default=None, type=str)
    # For MLP
    parser.add_argument("--z_dim", default=256, type=int)
    # For dropout
    parser.add_argument("--prob", default=0.5, type=float)
    parser.add_argument("--num_mask", default=1, type=int)

    args = parser.parse_args()
    if args.normal:
        assert args.normal_eval
    else:
        assert not args.normal_eval

    train(args)
