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
from kornia.augmentation import RandomErasing

from linear_models import Encoder, VectorQuantizer, weight_init
from utils import (
    load_dataset,
    evaluate_crlr,
    set_seed_everywhere,
    categorical_confounder_balancing_loss,
)
from dopamine.discrete_domains.atari_lib import create_atari_environment
from sklearn.linear_model import LogisticRegression


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

    save_dir = "models_vqvae_cnn_actor_crlr"

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

    for p in encoder.parameters():
        p.requires_grad = False
    for p in quantizer.parameters():
        p.requires_grad = False
    vqvae_dict = torch.load(args.vqvae_path, map_location="cpu")
    encoder.load_state_dict(
        {k[9:]: v for k, v in vqvae_dict.items() if "_encoder" in k}
    )
    quantizer.load_state_dict(
        {k[11:]: v for k, v in vqvae_dict.items() if "_quantizer" in k}
    )

    ## Multi-GPU
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        quantizer = nn.DataParallel(quantizer)

    criterion = nn.CrossEntropyLoss()
    logging.info("Training starts..")
    f_tr = open(os.path.join(save_dir, save_tag + "_cnn_train.csv"), "w")
    writer_tr = csv.writer(f_tr)
    writer_tr.writerow(["Epoch", "Actor Loss", "Weight Loss", "Accuracy"])

    f_te = open(os.path.join(save_dir, save_tag + "_cnn_eval.csv"), "w")
    writer_te = csv.writer(f_te)
    writer_te.writerow(["Epoch", "Actor Loss", "Weight Loss", "Accuracy", "Score"])

    if args.idx_path is None:
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
        if not os.path.exists("./total_idx"):
            os.makedirs("./total_idx")
        torch.save(
            total_encoding_indices,
            os.path.join("./total_idx", save_tag + "_total_idx.pth"),
        )
    else:
        total_encoding_indices = torch.load(args.idx_path, map_location="cpu")

    N, P = total_encoding_indices.shape
    total_encoding_onehot = torch.zeros(
        (N * P, args.num_embeddings), device=total_encoding_indices.device
    )
    total_encoding_onehot.scatter_(
        1, total_encoding_indices.reshape(-1).unsqueeze(1), 1
    )
    total_encoding_onehot = total_encoding_onehot.view(N, P, args.num_embeddings)  # NPE

    actor = nn.Linear(args.num_embeddings * P, action_dim,).to(device)
    if torch.cuda.device_count() > 1:
        actor = nn.DataParallel(actor)

    criterion = nn.CrossEntropyLoss(reduction="none")
    total_actions = torch.as_tensor(actions, device=device).long()

    x_total = torch.flatten(total_encoding_onehot, start_dim=1).detach()  # ND
    y_total = total_actions.detach()  # N
    x_total_np = x_total.cpu().numpy()
    y_total_np = y_total.cpu().numpy()
    if args.fixed_size is None:
        fixed_size = len(x_total)
    else:
        fixed_size = args.fixed_size

    if len(x_total) > fixed_size:
        weight = torch.full(
            [fixed_size], 1.0 / fixed_size, requires_grad=True, device=device
        )
        proj = torch.eye(fixed_size) - torch.ones(fixed_size, fixed_size) / fixed_size
        proj = proj.to(device)

        sample_idx = np.random.choice(len(x_total), fixed_size)
        x_total = x_total[sample_idx].to(device)
        y_total = y_total[sample_idx].to(device)
        x_total_np = x_total_np[sample_idx]
        y_total_np = y_total_np[sample_idx]
        total_encoding_indices = total_encoding_indices[sample_idx].to(device)
        total_encoding_onehot = total_encoding_onehot[sample_idx].to(device)
        total_actions = total_actions[sample_idx]
    else:
        weight = torch.full([N], 1.0 / N, requires_grad=True, device=device)
        proj = torch.eye(N) - torch.ones(N, N) / N
        proj = proj.to(device)

    for epoch in tqdm(range(args.n_epochs)):
        actor_losses = []
        weight_losses = []
        accuracies = []
        sample_weight = weight.detach().cpu().numpy()  # N
        actor_clf = LogisticRegression(random_state=args.seed, n_jobs=-1).fit(
            x_total_np, y_total_np, sample_weight=sample_weight
        )
        cls_list = actor_clf.classes_
        if not ((max(cls_list) == len(cls_list) - 1) and (min(cls_list) == 0)):
            raise ValueError("class re-mapping is needed")
        if torch.cuda.device_count() > 1:
            actor.module.weight.data = (
                torch.from_numpy(actor_clf.coef_).float().to(device)
            )
            actor.module.bias.data = (
                torch.from_numpy(actor_clf.intercept_).float().to(device)
            )
        else:
            actor.weight.data = torch.from_numpy(actor_clf.coef_).float().to(device)
            actor.bias.data = torch.from_numpy(actor_clf.intercept_).float().to(device)
        with torch.no_grad():
            logits = actor(x_total)

        for ii in tqdm(range(args.num_sub_iters)):
            weight_loss = categorical_confounder_balancing_loss(
                total_encoding_indices,
                weight,
                args.num_embeddings,
                total_encoding_onehot,
            )
            actor_loss = criterion(logits, total_actions)
            loss = weight @ actor_loss.detach() + args.lmd * weight_loss
            loss.backward()
            with torch.no_grad():
                weight -= args.lr * (proj @ weight.grad)
                weight.abs_()  ## non-negative weight
                weight /= weight.sum()  ## normalization
                weight.grad.zero_()

        accuracy = (total_actions == logits.argmax(1)).float().mean()
        actor_losses.append(actor_loss.mean().detach().cpu().item())
        weight_losses.append(weight_loss.mean().detach().cpu().item())
        accuracies.append(accuracy.mean().detach().cpu().item())

        logging.info(
            "Epochs {} | Actor Loss: {:.4f} | Weight Loss: {:.4f} | Accuracy: {:.2f}".format(
                epoch + 1,
                np.mean(actor_losses),
                np.mean(weight_losses),
                np.mean(accuracies),
            )
        )
        writer_tr.writerow(
            [
                epoch + 1,
                np.mean(actor_losses),
                np.mean(weight_losses),
                np.mean(accuracies),
            ]
        )

        if (epoch + 1) % args.eval_interval == 0:
            actor.eval()
            encoder.eval()
            quantizer.eval()
            score = evaluate_crlr(
                env,
                actor.module if torch.cuda.device_count() > 1 else actor,
                encoder.module if torch.cuda.device_count() > 1 else encoder,
                encoder.module if torch.cuda.device_count() > 1 else encoder,
                quantizer.module if torch.cuda.device_count() > 1 else quantizer,
                device,
                args,
            )
            logging.info("(Eval) Epoch {} | Score: {:.2f}".format(epoch + 1, score,))
            actor.train()
            writer_te.writerow(
                [
                    epoch + 1,
                    np.mean(actor_losses),
                    np.mean(weight_losses),
                    np.mean(accuracies),
                    score,
                ]
            )

    f_tr.close()
    f_te.close()

    torch.save(
        actor.module.state_dict()
        if (torch.cuda.device_count() > 1)
        else actor.state_dict(),
        os.path.join(save_dir, save_tag + "_ep{}_actor.pth".format(epoch + 1),),
    )
    torch.save(
        weight, os.path.join(save_dir, save_tag + "_ep{}_weight.pth".format(epoch + 1)),
    )
    if len(x_total) > fixed_size:
        torch.save(
            sample_idx,
            os.path.join(save_dir, save_tag + "_ep{}_sample_idx.pth".format(epoch + 1)),
        )

    torch.save(
        encoder.module.state_dict()
        if torch.cuda.device_count() > 1
        else encoder.state_dict(),
        os.path.join(save_dir, save_tag + "_ep{}_encoder.pth".format(epoch + 1)),
    )
    torch.save(
        quantizer.module.state_dict()
        if torch.cuda.device_count() > 1
        else quantizer.state_dict(),
        os.path.join(save_dir, save_tag + "_ep{}_quantizer.pth".format(epoch + 1)),
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
    # For CRLR
    parser.add_argument("--lmd", default=1e-1, type=float)
    parser.add_argument("--num_sub_iters", default=50, type=int)
    parser.add_argument("--fixed_size", default=None, type=int)
    parser.add_argument("--idx_path", default=None, type=str)

    args = parser.parse_args()
    if args.normal:
        assert args.normal_eval
    else:
        assert not args.normal_eval

    train(args)
