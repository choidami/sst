from __future__ import division
from __future__ import print_function

import json
import itertools
import argparse
import math
import time
import argparse
import pickle
import os
from functools import partial
import numpy as np

import torch
torch.set_printoptions(precision=32)
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler

from modules import MLPEncoder, MLPDecoder
from utils import get_experiments_folder, get_experiment_name
from utils import load_data, encode_onehot
from utils import maybe_make_logits_symmetric
from utils import nll_gaussian, kl_categorical_uniform, kl_gumbel
from utils import sampling_edge_metrics
from utils import sample_indep_edges
from core.spanning_tree import sample_tree_from_logits
from core.topk import sample_topk_from_logits

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=eval, default=True, choices=[True, False],
                    help="Enables CUDA training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

parser.add_argument("--mode", type=str, default="train", 
                    choices=["train", "eval"],
                    help="Whether to train or evaluate.")
parser.add_argument("--num_iterations", type=int, default=50000,
                    help="Number of iterations to train.")
parser.add_argument("--eval_every", type=int, default=500,
                    help="Number of training steps in-between evaluating.")
parser.add_argument("--batch_size", type=int, default=128,
                    help="Number of samples per batch.")
parser.add_argument("--eval_batch_size", type=int, default=100,
                    help="Number of samples per batch for eval on validation.")
parser.add_argument("--temp", type=float, default=5.0, help="Temperature.")
parser.add_argument("--lr", type=float, default=0.0003,
                    help="Initial learning rate.")
parser.add_argument("--lr_decay", type=int, default=200,
                    help="After how epochs to decay LR by a factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="LR decay factor.")
parser.add_argument("--enc_weight_decay", type=float, default=0.0,
                    help="Weight decay for AdamW.")
parser.add_argument("--dec_weight_decay", type=float, default=0.0,
                    help="Weight decay for AdamW.")

parser.add_argument("--encoder_hidden", type=int, default=256,
                    help="Number of hidden units.")
parser.add_argument("--decoder_hidden", type=int, default=256,
                    help="Number of hidden units.")
parser.add_argument("--num_vertices", type=int, default=10,
                    help="Number of vertices in the graph.")
parser.add_argument("--encoder_dropout", type=float, default=0.0,
                    help="Dropout rate (1 - keep probability).")
parser.add_argument("--decoder_dropout", type=float, default=0.0,
                    help="Dropout rate (1 - keep probability).")
parser.add_argument("--factor", type=eval, default=True,
                    choices=[True, False],
                    help="Enables factor graph model.")

parser.add_argument("--suffix", type=str, default="_novar_1skip_10t_1r_graph10",
                    help="Suffix for training data.")
parser.add_argument("--edge_types", type=int, default=2, choices=[1, 2],
                    help="The number of edge types to infer. Must be <= 2.")
parser.add_argument("--dims", type=int, default=2,
                    help="The number of input dimensions.")
parser.add_argument("--timesteps", type=int, default=10,
                    help="The number of time steps per sample.")
parser.add_argument("--prediction_steps", type=int, default=10, metavar="N",
                    help="Num steps to predict before re-using teacher forcing.")
parser.add_argument("--num_rounds", type=int, default=1,
                    help="Num message passing rounds in decoder per timestep.")

parser.add_argument("--skip_first", type=eval, default=False, choices=[True, False],
                    help="Skip first edge type in decoder, i.e. it represents no-edge.")
parser.add_argument("--var", type=float, default=5e-5, help="Output variance.")
parser.add_argument("--hard", type=eval, default=False, choices=[True, False],
                    help="Uses discrete samples in training forward pass.")
parser.add_argument("--st", type=eval, default=False, choices=[True, False],
                    help="Uses discrete samples in training forward pass.")

parser.add_argument("--sst", type=str, default="tree",
                    choices=["indep", "tree", "topk"],
                    help="Stochastic Softmax Tricks")
parser.add_argument("--relaxation", type=str, default="exp_family_entropy",
                    help="Relaxation for SST.") 
parser.add_argument("--max_range", type=float, default=np.inf,
                    help="Max range of logits for spanning tree sst.")
parser.add_argument("--eps_for_finitediff", type=float, default=1e-2,
                    help="Epsilon for finite difference for topk.")
parser.add_argument("--use_gumbels_for_kl", type=eval, default=True,
                    choices=[True, False],
                    help="Whether to compute KL wrt U (gumbels) instead of X.")

parser.add_argument("--use_nvil", type=eval, default=False,
                    choices=[True, False], help="Whether to use NVIL.")
parser.add_argument("--use_reinforce", type=eval, default=False,
                    choices=[True, False], help="Whether to use REINFORCE.")
parser.add_argument("--num_samples", type=int, default=1,
                    help="Num. samples for gradient estimation.")
parser.add_argument("--reinforce_baseline", type=str, default="ema",
                    choices=["ema", "batch", "multi_sample"],
                    help="Choice of baseline for REINFORCE.")
parser.add_argument("--ema_for_loss", type=float, default=0.99,
                    help="EMA coefficient for NVIL or REINFORCE.")

parser.add_argument("--use_cpp_for_sampling", type=eval, default=True,
                    choices=[True, False], 
                    help=("Whether to use C++ Kruskal's when sampling for "
                          "spanning tree sst."))
parser.add_argument("--use_cpp_for_edge_metric", type=eval, default=False,
                    choices=[True, False],
                    help=("Whether to use C++ Kruskal's when computing edge "
                          "metrics for spanning tree sst."))
parser.add_argument("--edge_metric_num_samples", type=int, default=1,
                    help="Num. samples when computing edge metrics.")
parser.add_argument("--log_edge_metric_train", type=eval, default=False,
                    choices=[True, False], 
                    help="Whether to compute and log edge metrics on train.")
parser.add_argument("--log_edge_metric_val", type=eval, default=True,
                    choices=[True, False],
                    help="Whether to compute and log edge metrics on valid.")
parser.add_argument("--eval_edge_metric_bs", type=int, default=10000,
                    help="Batch size for computing edge metrics during eval.")
parser.add_argument("--symmeterize_logits", type=eval, default=True,
                    choices=[True, False],
                    help="Whether to make the encoder output edge symmetric.")

parser.add_argument("--experiments_folder", type=str, default=None,
                    help=("Name of folder for experiment group."
                          "Set this for evaluation (mode == 'eval'."))
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Name of experiment.")
parser.add_argument("--save_best_model", type=eval, default=True,
                    choices=[True, False],
                    help="Whether to save the checkpoint for the best model.")
parser.add_argument("--add_timestamp", type=eval, default=True, 
                    choices=[True, False],
                    help="Whether to add timestamp to experiments folder.")
args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if args.cuda else "cpu"

# Check arguments.
if args.sst != "indep":
    assert args.use_gumbels_for_kl
if args.use_nvil or args.use_reinforce:
    assert not (args.use_nvil and args.use_reinforce)
    assert args.hard
if args.use_reinforce and args.reinforce_baseline == "multi_sample":
    assert args.num_samples > 1
if args.mode == "eval":
    assert args.experiments_folder is not None

if args.mode == "train":
    # Experiments are organized such that there is a main experiments folder
    # which contains experiments that share the same training configuration
    # (same SST, relaxation, etc...) except for the hyperparameters.
    # First, set up main experiments folder.
    experiments_folder = (args.experiments_folder if args.experiments_folder 
                        else get_experiments_folder(args))
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    if not os.path.exists(os.path.join("experiments", experiments_folder)):
        os.makedirs(os.path.join("experiments", experiments_folder))
    # Set up the folder for specific hyperparameter settings.
    experiment_name = get_experiment_name(args)
    experiment_folder = os.path.join(
        "experiments", experiments_folder, experiment_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    # Save args in experiment folder.
    with open(os.path.join(experiment_folder, "train_config.json"), "w") as f:
        config = {k: v for (k, v) in vars(args).items()}
        json.dump(config, f, indent=2)

    # Get ready to save model.
    encoder_file = os.path.join(experiment_folder, "encoder.pt")
    decoder_file = os.path.join(experiment_folder, "decoder.pt")
    log_file = os.path.join(experiment_folder, "log.txt")
    log = open(log_file, "w")

# Setup up training, validation, and test data.
train_loader, valid_loader, test_loader, num_train, num_valid, num_test = load_data(
    args.batch_size, args.eval_batch_size, args.suffix)
num_complete_batches, leftover = divmod(num_train, args.batch_size)
num_batches_per_epoch = num_complete_batches + bool(leftover)
# Make sure eval batch size divides validation set, since we assume this
# when computing eval metrics.
eval_edge_metric_bs = (args.eval_batch_size if not args.eval_edge_metric_bs 
                       else args.eval_edge_metric_bs)
assert num_valid % args.eval_batch_size == 0
assert num_valid % eval_edge_metric_bs == 0
if args.mode == "eval":
    assert num_test % args.eval_batch_size == 0
    assert num_test % eval_edge_metric_bs == 0

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_vertices, args.num_vertices]) - np.eye(args.num_vertices)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                     (args.edge_types if args.sst == "indep" else 1),
                     args.encoder_dropout, args.factor,
                     args.use_nvil, num_edges=rel_rec.size(0), n=args.num_vertices,
                     num_timesteps=args.timesteps, num_dims=args.dims)

decoder = MLPDecoder(n_in_node=args.dims,
                     edge_types=args.edge_types,
                     msg_hid=args.decoder_hidden,
                     msg_out=args.decoder_hidden,
                     n_hid=args.decoder_hidden,
                     do_prob=args.decoder_dropout,
                     skip_first=args.skip_first,
                     num_rounds=args.num_rounds)

if args.enc_weight_decay > 0.0 or args.dec_weight_decay > 0.0:
    print("Using AdamW.")
    optimizer = optim.AdamW([
        {"params": encoder.parameters(), "weight_decay": args.enc_weight_decay},
        {"params": decoder.parameters(), "weight_decay": args.dec_weight_decay}],
        lr = args.lr)
else:
    print("Using Adam.")
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Setup sampling function and probability calculation function for tree prior.
if args.sst == "tree":
    # Check that the output of the encode is made to be symmetric.
    assert args.symmeterize_logits is not None
    sample_edges = partial(sample_tree_from_logits, edge_types=args.edge_types,
                           relaxation=args.relaxation, max_range=args.max_range, 
                           use_cpp=args.use_cpp_for_sampling)
elif args.sst == "topk":
    # Check that the output of the encode is made to be symmetric.
    assert args.symmeterize_logits is not None
    sample_edges = partial(sample_topk_from_logits, k=(args.num_vertices - 1),
                           edge_types=args.edge_types, relaxation=args.relaxation,
                           eps=args.eps_for_finitediff)
elif args.sst == "indep":
    # sample_edges = gumbel_softmax
    sample_edges = partial(
        sample_indep_edges, is_edgesymmetric=args.symmeterize_logits)

else:
    raise ValueError(f"Stochastic Softmax Trick type {args.sst} is not valid!")

def compute_kl(logits):
    if args.sst == "indep" and not args.use_gumbels_for_kl:
        probs = F.softmax(logits, dim=-1)
        return kl_categorical_uniform(probs, args.num_vertices, args.edge_types)
    else:
        return kl_gumbel(logits, args.num_vertices)

get_sampling_metrics = partial(
    sampling_edge_metrics, sst=args.sst, n=args.num_vertices,
    num_samples=args.edge_metric_num_samples, 
    is_edgesymmetric=args.symmeterize_logits, use_cpp=args.use_cpp_for_edge_metric)

if args.cuda:
    encoder.to("cuda")
    decoder.to("cuda")
    rel_rec = rel_rec.to("cuda")
    rel_send = rel_send.to("cuda")

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train():
    itercount = itertools.count()
    num_epochs = math.ceil(args.num_iterations / num_batches_per_epoch)

    best_elbo = -np.inf
    best_step, best_epoch = 0, 0

    # Exponential moving average of negative log-likelihood for 
    # NVIL and REINFORCE.
    loss_nll_ema = 0.0

    measurements = {
        # Measurements on training set.
        "train_steps": [], "nll_train": [], 
        "acc_train": [], "precision_train": [], "recall_train": [],
        "elbo_train":[], "tf_elbo_train": [], "kl_train": [], 
        "mse_train": [],
        # Measurements on validation set.
        "val_steps": [], "nll_val": [], 
        "acc_val": [], "precision_val": [], "recall_val": [],
        "elbo_val": [], "kl_val": [], "mse_val": [],
    }

    encoder.train()
    decoder.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        for batch_idx, (data, relations) in enumerate(train_loader):
            i = next(itercount)
            if args.cuda:
                data, relations = data.to("cuda"), relations.to("cuda")
            data, relations = Variable(data), Variable(relations)

            optimizer.zero_grad()

            logits, nvil_baseline = encoder(data, rel_rec, rel_send)
            
            logits = maybe_make_logits_symmetric(logits, args.symmeterize_logits)
            
            edges = []
            edge_weights = []
            for _ in range(args.num_samples):
                # ss stands for single sample.
                ss_edges, ss_edge_weights = sample_edges(
                    logits, tau=args.temp, hard=args.hard, hard_with_grad=args.st)
                edges.append(ss_edges)
                edge_weights.append(ss_edge_weights)
            # Edges and edge_weights are of shape 
            # (num_samples * bs, (n - 1) * n, edge_types).
            edges = torch.cat(edges)  
            edge_weights = torch.cat(edge_weights)

            if args.use_nvil or args.use_reinforce:
                edges = edges.detach()
                edge_weights = edge_weights.detach()
            
            # Repeat data to account for multiple samples.
            data = data.repeat(
                args.num_samples, *([1] * len(data.shape[1:])))
            
            output = decoder(data, edges, rel_rec, rel_send, args.prediction_steps)

            target = data[:, :, 1:, :]

            loss_nll = nll_gaussian(output, target, args.var)
            # Reshape to take into account num_samples.
            loss_nll = loss_nll.view(args.num_samples, logits.size(0))
            # Unsqueeze to take into account num_samples.
            loss_kl = compute_kl(logits).unsqueeze(0)
            # Make sure all losses are consistently divided by num_vertices.
            loss = (loss_nll + loss_kl).sum() / (logits.size(0) * args.num_samples)

            measurements["train_steps"].append(i)
            if args.log_edge_metric_train:
                acc, precision, recall = get_sampling_metrics(logits, relations)
                best_idx = np.argmax(precision)
                measurements["acc_train"].append(acc[best_idx])
                measurements["precision_train"].append(precision[best_idx])
                measurements["recall_train"].append(recall[best_idx])
            mse_loss = F.mse_loss(output, target).item()
            measurements["mse_train"].append(mse_loss)
            measurements["nll_train"].append(loss_nll.mean().item())
            measurements["kl_train"].append(loss_kl.mean().item())
            measurements["elbo_train"].append(-1.0 * loss.item())

            # Get decoder output using teacher forcing. We do this to get a 
            # comparable elbo measurement with evaluation results, since we use
            # teacherforcing when evaluating on the validation and test set.
            with torch.no_grad():
                # Here, tf stands for teacher forcing.
                tf_output = decoder(data, edges, rel_rec, rel_send, 1)
                tf_nll = nll_gaussian(tf_output, target, args.var)
                tf_nll = tf_nll.view(args.num_samples, logits.size(0))
                tf_loss = (tf_nll + loss_kl).sum() / (logits.size(0) * args.num_samples)
                tf_elbo = -1.0 * tf_loss
                measurements["tf_elbo_train"].append(tf_elbo.item())

            if args.use_nvil or args.use_reinforce:
                # Compute log p with respect to U.
                edge_weights = edge_weights.view(args.num_samples, *logits.shape)
                if args.use_gumbels_for_kl:
                    logprob = (
                        -(edge_weights - logits.unsqueeze(0)) - 
                        torch.exp(-(edge_weights - logits.unsqueeze(0))))
                else:
                    # Compute log p with respect to X. This only makes sense
                    # when args.sst == 'indep'.
                    edges = edges.view(args.num_samples, *logits.shape)
                    logprob = torch.log(torch.sum(
                        F.softmax(logits, dim=-1).unsqueeze(0) * edges, 
                        axis=-1, keepdim=True))
                logprob = logprob.sum(-1).sum(-1)

                # Exponential moving average on the loss.
                # If ema coeff is 0 then baseline is also 0.
                if args.ema_for_loss > 0.0:
                    loss_nll_ema = (
                        args.ema_for_loss * loss_nll_ema + 
                        (1.0 - args.ema_for_loss) * loss_nll.mean()).detach()
                else:
                    loss_nll_ema = 0.0
            
                if args.use_nvil:
                    nvil_baseline = nvil_baseline.unsqueeze(0)
                    baseline_loss = ((
                        (loss_nll - loss_nll_ema).detach() - nvil_baseline) ** 2)
                    nvil_loss = (
                        loss_nll +
                        (loss_nll - loss_nll_ema - nvil_baseline).detach() * logprob + 
                        baseline_loss / args.num_vertices
                    )
                    nvil_loss = (nvil_loss + loss_kl).sum() / (logits.size(0) * args.num_samples)

                    nvil_loss.backward()
                    optimizer.step()
                else:  # REINFORCE
                    # Compute the baseline.
                    if args.reinforce_baseline == "ema":
                        baseline = loss_nll_ema
                    elif args.reinforce_baseline == "batch":
                        # Use the mean of the whole batch.
                        # Compute mean over each sample separately.
                        baseline = loss_nll.mean()
                    elif args.reinforce_baseline == "multi_sample":
                        baseline = loss_nll.mean(0).unsqueeze(0)  # (1, bs)

                    reinforce_loss = loss_nll + (loss_nll - baseline).detach() * logprob
                    # Divide by (num_samples - 1) in the multi-sample case for
                    # an unbiased estimate.
                    reinforce_loss = reinforce_loss.sum(0) / (
                        (args.num_samples - 1) if args.reinforce_baseline == "multi_sample"
                        else args.num_samples
                    )
                    reinforce_loss = (reinforce_loss + loss_kl).sum() / logits.size(0)

                    reinforce_loss.backward()
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()

            # Evaluate every args.eval_every steps.
            if i % args.eval_every == 0:
                train_time = time.time() - start_time
                start_time = time.time()
                eval_start_time = time.time()

                measurements["val_steps"].append(i)
                nlls, mses = [], []
                accs, precisions, recalls = [], [], []
                kls, elbos = [], []
                logits_list_for_eval, relations_list_for_eval = [], []

                encoder.eval()
                decoder.eval()
                for batch_idx, (data, relations) in enumerate(valid_loader):
                    if args.cuda:
                        data, relations = data.to("cuda"), relations.to("cuda")
                    data, relations = Variable(data), Variable(relations)

                    logits, baseline = encoder(data, rel_rec, rel_send)
                    logits = maybe_make_logits_symmetric(logits, args.symmeterize_logits)
                    edges, _ = sample_edges(logits, tau=args.temp, hard=True)

                    # validation output uses teacher forcing.
                    output = decoder(data, edges, rel_rec, rel_send, 1)

                    target = data[:, :, 1:, :]
                    loss_nll = nll_gaussian(output, target, args.var).mean()
                    loss_kl = compute_kl(logits).mean()
                    
                    # Since computing the edge metrics can be done with a
                    # much bigger batch size the eval batch size (for obtaining
                    # encoder and decoder outputs), we might want to collect
                    # the encoder output logits and compute edge metics
                    # with a bigger batch size outside the eval loop.
                    if args.log_edge_metric_val and eval_edge_metric_bs == args.eval_batch_size:
                        acc, precision, recall = get_sampling_metrics(logits, relations)
                        accs.append(acc)
                        precisions.append(precision)
                        recalls.append(recall)
                    elif args.log_edge_metric_val and eval_edge_metric_bs != args.eval_batch_size:
                        logits_list_for_eval.append(logits.to("cpu").detach().numpy())
                        relations_list_for_eval.append(relations.to("cpu").detach().numpy())

                    mses.append(F.mse_loss(output, target).item())
                    nlls.append(loss_nll.item())
                    kls.append(loss_kl.item())
                    elbos.append(-1.0 * (loss_nll + loss_kl).item())

                # Compute edge metrics with a bigger batch size separately
                # from the eval loop. For spanning tree SST, doing this is
                # faster only when the batched pytorch version of Kruskal's is
                # used. The C++ Kruskal's is faster for small batch sizes
                # (for example, when evaluating a bigger graph where we can
                # only fit a small batch size for the encoder and decoder.)
                if args.log_edge_metric_val and eval_edge_metric_bs != args.eval_batch_size:
                    logits_for_eval = torch.tensor(np.vstack(logits_list_for_eval)).to(device)
                    relations_for_eval = torch.tensor(np.vstack(relations_list_for_eval)).to(device)
                    
                    for sub_idx in range(int(logits_for_eval.size(0) / eval_edge_metric_bs)):
                        logits_ = logits_for_eval[
                            sub_idx * eval_edge_metric_bs: (sub_idx + 1) * eval_edge_metric_bs]
                        relations_ = relations_for_eval[
                            sub_idx * eval_edge_metric_bs: (sub_idx + 1) * eval_edge_metric_bs]
                        acc, precision, recall = get_sampling_metrics(logits_, relations_)
                        accs.append(acc)
                        precisions.append(precision)
                        recalls.append(recall)
                        
                measurements["nll_val"].append(np.mean(nlls))
                measurements["kl_val"].append(np.mean(kls))
                measurements["elbo_val"].append(np.mean(elbos))
                measurements["mse_val"].append(np.mean(mses))
                accs = np.mean(accs, axis=0)
                precisions = np.mean(precisions, axis=0)
                recalls = np.mean(recalls, axis=0)
                best_idx = np.argmax(precisions)
                measurements["acc_val"].append(accs[best_idx])
                measurements["precision_val"].append(precisions[best_idx])
                measurements["recall_val"].append(recalls[best_idx])

                eval_time = time.time() - eval_start_time
                print(
                    "{}/{} iterations in {:0.2f}s; ".format(
                        i, args.num_iterations, train_time) +
                    "Eval in {:0.2f} sec".format(eval_time), flush=True)
                measurements_str = (
                    "Iteration {} (Epoch {}) ".format(i, epoch) +
                    "nll_train: {:.10f} ".format(measurements["nll_train"][-1]) +
                    "kl_train: {:.10f} ".format(measurements["kl_train"][-1]) +
                    "elbo_train: {:.10f} ".format(measurements["elbo_train"][-1]) +
                    "tf_elbo_train: {:.10f} ".format(measurements["tf_elbo_train"][-1]) +
                    "mse_train: {:.10f} ".format(measurements["mse_train"][-1]) +
                    ("acc_train: {:.10f} ".format(measurements["acc_train"][-1]) + 
                     "precision_train: {:.10f} ".format(measurements["precision_train"][-1]) +
                     "recall_train: {:.10f} ".format(measurements["recall_train"][-1])
                     if args.log_edge_metric_train else "") +
                    "nll_val: {:.10f} ".format(measurements["nll_val"][-1]) +
                    "kl_val: {:.10f} ".format(measurements["kl_val"][-1]) +
                    "elbo_val: {:.10f} ".format(measurements["elbo_val"][-1]) +
                    "mse_val: {:.10f} ".format(measurements["mse_val"][-1]) +
                    ("acc_val: {:.10f} ".format(measurements["acc_val"][-1]) +
                     "precision_val: {:.10f} ".format(measurements["precision_val"][-1]) +
                     "recall_val: {:.10f} ".format(measurements["recall_val"][-1])
                     if args.log_edge_metric_val else "")
                )
                print(measurements_str)
                print(measurements_str, file=log)
                log.flush()

                if args.save_best_model and measurements["elbo_val"][-1] > best_elbo:
                    torch.save(encoder.state_dict(), encoder_file)
                    torch.save(decoder.state_dict(), decoder_file)
                    print("Best model so far, saving...")

                if measurements["elbo_val"][-1] > best_elbo:
                    best_elbo = measurements["elbo_val"][-1]
                    best_step, best_epoch = i, epoch

                encoder.train()
                decoder.train()

        scheduler.step()

    return measurements, best_elbo, best_step, best_epoch


def test():
    measurements = {
        "valid":{
            "elbo": [], "acc": [], "precision": [], "recall": []},
        "test":{
            "elbo": [], "acc": [], "precision": [], "recall": []}
    }
    idx = 0
    for exp in os.listdir(os.path.join("experiments", args.experiments_folder)):
        trial_path = os.path.join("experiments", args.experiments_folder, exp)
        if not os.path.os.path.isdir(trial_path):
            continue
        if not os.path.exists(os.path.join(trial_path, "train_and_val_measurements.pkl")):
            continue
        start = time.time()
        try:
            encoder_file = os.path.join(trial_path, "encoder.pt")
            encoder.load_state_dict(
                torch.load(encoder_file, map_location=torch.device(device)))
            decoder_file = os.path.join(trial_path, "decoder.pt")
            decoder.load_state_dict(
                torch.load(decoder_file, map_location=torch.device(device)))
        except:
            continue
        
        for dataset in ["valid", "test"]:
            dataloader = valid_loader if dataset == "valid" else test_loader
            elbos = []
            logits_list = []
            relations_list = []
            accs_list, precisions_list, recalls_list = [], [], []
            for batch_idx, (data, relations) in enumerate(dataloader):
                data = data[:, :, :args.timesteps, :]
                if args.cuda:
                    data, relations = data.to("cuda"), relations.to("cuda")
                data, relations = Variable(data), Variable(relations)

                logits, _ = encoder(data, rel_rec, rel_send)
                logits = maybe_make_logits_symmetric(logits, args.symmeterize_logits)
                edges, _ = sample_edges(logits, tau=args.temp, hard=True)

                # validation output uses teacher forcing.
                output = decoder(data, edges, rel_rec, rel_send, 1)

                target = data[:, :, 1:, :]
                loss_nll = nll_gaussian(output, target, args.var).mean()
                loss_kl = compute_kl(logits).mean()

                elbos.append(-1.0 * (loss_nll + loss_kl).item())
                logits_list.append(logits.to("cpu").detach().numpy())
                relations_list.append(relations.to("cpu").detach().numpy())

            logits_for_eval = torch.tensor(np.vstack(logits_list)).to(device)
            relations_for_eval = torch.tensor(np.vstack(relations_list)).to(device)
            for sub_idx in range(int(logits_for_eval.size(0) / eval_edge_metric_bs)):
                logits_ = logits_for_eval[
                    sub_idx * eval_edge_metric_bs: (sub_idx + 1) * eval_edge_metric_bs]
                relations_ = relations_for_eval[
                    sub_idx * eval_edge_metric_bs: (sub_idx + 1) * eval_edge_metric_bs]
                accs, precisions, recalls = get_sampling_metrics(logits_, relations_)
                accs_list.append(accs)
                precisions_list.append(precisions)
                recalls_list.append(recalls)
        
            print(f"{dataset} trial {idx} for {args.experiments_folder} took {time.time() - start}s.")
            measurements[dataset]["elbo"].append(np.mean(elbos))
            accs = np.mean(accs_list, axis=0)
            precisions = np.mean(precisions_list, axis=0)
            recalls = np.mean(recalls_list, axis=0)
            best_idx = np.argmax(precisions)
            measurements[dataset]["acc"].append(accs[best_idx])
            measurements[dataset]["precision"].append(precisions[best_idx])
            measurements[dataset]["recall"].append(recalls[best_idx])
        idx += 1

    all_measurements = {}
    for dataset in measurements:
        all_measurements[dataset] = {}
        for k, v in measurements[dataset].items():
            all_measurements[dataset][k] = np.array(v)
    return all_measurements


if args.mode == "train":
    # Train model
    train_and_val_measurements, best_elbo, best_step, best_epoch = train()
    print("Optimization Finished!")
    print("Best Epoch: {:04d}; best step: {:04d}".format(best_epoch, best_step))
    print("Best Epoch: {:04d}; best step: {:04d}".format(best_epoch, best_step), file=log)
    log.flush()

    # Save measurements.
    meas_fname = "train_and_val_measurements.pkl"
    with open(os.path.join(experiment_folder, meas_fname), "wb") as f:
        pickle.dump(train_and_val_measurements, f)

else:
    all_measurements = test()
    import pdb;pdb.set_trace()
    print("Saving data.")
    data_path = os.path.join(
        "experiments", args.experiments_folder, "data_for_bootstrapping.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(all_measurements, f)
