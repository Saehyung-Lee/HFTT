import datetime
import random
import statistics
import sys
import math
import os
import time
import warnings
import copy
import pickle

import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from typing import Dict, Callable, Optional, Any, Tuple, Union
import numpy as np
import statistics
import logging
import torch
import torch.utils.data
from torch.nn.functional import normalize
from model import get_model
import clip
from imagenet import imagenet_templates, imagenet_classes
from tqdm import tqdm
import torch.nn.functional as F

from transformers import AutoProcessor, BlipForImageTextRetrieval
from sklearn.metrics import roc_auc_score
from dataset import TextOOD, OOD

os.environ['TOKENIZERS_PARALLELISM']='true'

def get_zeroshot_classifier(model, classnames, templates, args, processor=None):
    with torch.no_grad():
        zeroshot_weights = []
        class_embeddings_all = {}
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            if 'clip' in args.model:
                texts = {'text': clip.tokenize(texts).cuda()} #tokenize
            elif 'blip' in args.model:
                texts = processor(text=texts, padding=True, return_tensors='pt')
            class_embeddings = normalize(model.encode_text(**texts), dim=-1) #embed with text encoder
            class_embeddings_all[classname] = class_embeddings
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights, torch.cat(list(class_embeddings_all.values()), dim=0)

def detection_loss(softmax, y_target, focal=0.):
    if focal > 0.:
        with torch.no_grad():
            coeff = torch.pow(1.-softmax[:,y_target[0]], focal)
            coeff = coeff / coeff.sum()
        return torch.sum(F.nll_loss(torch.log(softmax), y_target, reduction='none') * coeff)
    else:
        return F.nll_loss(torch.log(softmax), y_target)

def criterion(ind, ood, zeroshot_weights, device, args):
    num_id = zeroshot_weights.size(1) - args.num_ood_classes
    ind_targets = torch.zeros(len(ind), dtype=torch.long, device=device)
    ood_targets = torch.ones(len(ood), dtype=torch.long, device=device)
    ind_output = ind @ zeroshot_weights
    ind_p = F.softmax(ind_output / args.temperature, dim=-1)
    ind_p = torch.cat([ind_p[:,:num_id].sum(1, keepdim=True), ind_p[:,num_id:].sum(1, keepdim=True)], dim=1)
    ood_output = ood @ zeroshot_weights
    ood_p = F.softmax(ood_output / args.temperature, dim=-1)
    ood_p = torch.cat([ood_p[:,:num_id].sum(1, keepdim=True), ood_p[:,num_id:].sum(1, keepdim=True)], dim=1)
    return 0.5*detection_loss(ind_p, ind_targets) + 0.5*detection_loss(ood_p, ood_targets, args.focal)


def train_one_epoch(model, in_classifier, ood_classifier_, optimizer, in_text_features, data_loader, device, epoch, args, ind_features, OODs_features, processor=None):
    model.train()
    in_txts = in_text_features.cuda()
    for i, txts in enumerate(data_loader):
        start_time = time.time()

        zeroshot_classifier = torch.cat([in_classifier, normalize(ood_classifier_, dim=0)], dim=1)

        if args.num_eval_in_an_epoch > 0 and i % args.eval_freq == 0:
            auroc, fpr = 0, 0
            for k, v in OODs_features.items():
                logging.info("-"*10+f" OOD: {k} "+"-"*10)
                auroc_, fpr_ = evaluate_with_features(zeroshot_classifier, ind_features, v, device=device, args=args)
                auroc += auroc_
                fpr += fpr_
            auroc /= len(OODs_features)
            fpr /= len(OODs_features)
            logging.info(f">>>>> Average AUROC: {auroc} <<<<<")
            logging.info(f">>>>> Average FPR: {fpr} <<<<<")

        with torch.no_grad():
            if 'clip' in args.model:
                ood_txts = {'text': clip.tokenize(txts, truncate=True).to(device)}
            elif 'blip' in args.model:
                ood_txts = processor(text=txts, padding=True, return_tensors='pt')
            ood_txts = normalize(model.module.encode_text(**ood_txts), dim=-1)

        loss = criterion(in_txts, ood_txts, zeroshot_classifier, device, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            logging.info(f"Epoch: {epoch}, iter: {i}, Loss: {loss.item()}")
        if args.iters:
            if i == args.iters:
                break

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def evaluate(model, zeroshot_classifier, in_image_loader, ood_image_loader, device, args, log_suffix=""):
    model.eval()
    num_id = zeroshot_classifier.size(1) - args.num_ood_classes
    logits = []
    num_id_samples = 0
    with torch.inference_mode():
        for imgs, _ in in_image_loader:
            start_time = time.time()
            imgs = imgs.to(device)
            img_features = normalize(model.module.encode_image(imgs), dim=-1)
            logit = img_features @ zeroshot_classifier
            logits.append(logit.cpu())
            num_id_samples += len(imgs)
        for imgs, _ in ood_image_loader:
            start_time = time.time()
            imgs = imgs.to(device)
            img_features = normalize(model.module.encode_image(imgs), dim=-1)
            logit = img_features @ zeroshot_classifier
            logits.append(logit.cpu())
        logits = torch.cat(logits, dim=0)
        k = args.logit_coeff
        predictions = F.softmax(logits * k, dim=-1)
        if args.mode == 1:
            ood_scores = -torch.max(predictions, axis=1)[0]
        else:
            ood_scores = predictions[:,num_id:].sum(1)
        ood_targets = np.zeros(len(predictions), dtype=int)
        ood_targets[num_id_samples:] = 1
        auc = roc_auc_score(ood_targets, ood_scores)
        fpr = fpr_and_fdr_at_recall(ood_targets, ood_scores.numpy())
        logging.info(f"AUROC: {auc:.4f}")
        logging.info(f"FPR: {fpr:.4f}")
    return auc, fpr

def evaluate_with_features(zeroshot_classifier, ind_features, ood_features, device, args):
    num_id = zeroshot_classifier.size(1) - args.num_ood_classes
    logits = []
    num_id_samples = 0
    with torch.inference_mode():
        ind = ind_features.cuda()
        ood = ood_features.cuda()

        logit = ind @ zeroshot_classifier
        logits.append(logit.cpu())
        num_id_samples += len(logit)

        logit = ood @ zeroshot_classifier
        logits.append(logit.cpu())
        logits = torch.cat(logits, dim=0)
        k = args.logit_coeff
        predictions = F.softmax(logits * k, dim=-1)

        ood_scores = predictions[:,num_id:].sum(1)
        ood_targets = np.zeros(len(predictions), dtype=int)
        ood_targets[num_id_samples:] = 1
        auc = roc_auc_score(ood_targets, ood_scores)
        fpr = fpr_and_fdr_at_recall(ood_targets, ood_scores.numpy())
        logging.info(f"AUROC: {auc:.4f}")
        logging.info(f"FPR: {fpr:.4f}")
    return auc, fpr

def get_features(model, loader, device, args):
    model.eval()
    features = []
    with torch.inference_mode():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            img_features = normalize(model.module.encode_image(imgs), dim=-1)
            features.append(img_features.cpu())
        features = torch.cat(features, dim=0)
    return features

def load_data(args, processor):

    ind_name = args.ind.split('/')[-1]
    if 'blip' in args.model:
        image_processor = processor
        processor = lambda img: image_processor(images=img, return_tensors='pt')['pixel_values'][0]

    ''' in-distribution dataset '''
    in_image_dataset = torchvision.datasets.ImageFolder(os.path.join(args.ind, 'raw-data', 'val'), processor)
    in_classnames = imagenet_classes
    in_templates = imagenet_templates
    
    logging.info(f'# In-distribution: {len(in_image_dataset)}')

    ''' OOD dataset '''
    ood_text_dataset = TextOOD(args.ood_text_path)
    logging.info(f"# of OOD texts: {len(ood_text_dataset)}")

    if args.num_eval_in_an_epoch > 0:
        args.eval_freq = len(ood_text_dataset) // args.batch_size // args.num_eval_in_an_epoch
        args.eval_freq = 1 if args.eval_freq == 0 else args.eval_freq

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ood_text_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(ood_text_dataset)
    in_image_sampler = torch.utils.data.SequentialSampler(in_image_dataset)

    in_image_loader = torch.utils.data.DataLoader(
            in_image_dataset, batch_size=args.test_batch_size, sampler=in_image_sampler, num_workers=args.workers, pin_memory=True
    )
    ood_text_loader = torch.utils.data.DataLoader(
        ood_text_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=None
    )

    return ood_text_loader, in_image_loader, in_classnames, in_templates, train_sampler

def get_ood_image_loader(ood_path, processor):

    ood_name = ood_path.split('/')[-1]
    if 'blip' in args.model:
        image_processor = processor
        processor = lambda img: image_processor(images=img, return_tensors='pt')['pixel_values'][0]

    ''' out-distribution dataset '''
    ood_image_dataset = OOD(ood_path, transform=processor)
    logging.info(f'# Out-distribution: {len(ood_image_dataset)}')

    ood_image_sampler = torch.utils.data.SequentialSampler(ood_image_dataset)

    ood_image_loader = torch.utils.data.DataLoader(
            ood_image_dataset, batch_size=args.test_batch_size, sampler=ood_image_sampler, num_workers=args.workers, pin_memory=True
    )
    return ood_image_loader

def main(args):
    checkpoint = None
    if args.output_dir:
        utils.mkdir(args.output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    logging.info("Creating model")

    ''' EXP iter start '''
    OOD_evals = {}
    for exp in range(args.num_exp):
        logging.info('-'*10 + ' EXP %d '%exp + '-'*10)
        torch.manual_seed(args.seed+exp)
        np.random.seed(args.seed+exp)
        random.seed(args.seed+exp)

        model, processor = get_model(args)
        if args.mode == 'hftt' and 'blip' in args.model:
            model.set_ood_classifier(args.num_ood_classes)
        model.to(device)

        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.mode == 'hftt':
            for name, p in model.named_parameters():
                if 'ood_classifier_' == name:
                    print(f"update: {name}")
                    optim_param = p
                    p.requires_grad = True
                else:
                    print(f"not update: {name}")
                    p.requires_grad = False

        if exp == 0:
            ''' set data '''
            ood_text_loader, in_image_loader, in_classnames, in_templates, train_sampler = load_data(args, processor)
            OODs = {}
            ind_name = args.ind.split('/')[-1]
            for ood_name in ['sun397', 'inaturalist', 'places', 'dtd', 'NINCO/NINCO_OOD_classes', 'NINCO/NINCO_OOD_unit_tests']:
                OODs[ood_name] = get_ood_image_loader(os.path.join(args.ood, ood_name), processor)
                OOD_evals[ood_name] = {'aurocs': [], 'fprs': []}

            ''' set classifiers '''
            in_classifier, in_text_features = get_zeroshot_classifier(model, in_classnames, in_templates, args, processor)
        if args.mode == 'hftt':
            optimizer = torch.optim.SGD(
                [optim_param],
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr_min
            )
        else:
            optimizer = None
            args.epochs = 0
        ''''''

        logging.info(args)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        logging.info("Start training")
        start_time = time.time()
        best = 0
        best_fpr = 100
        if utils.is_main_process():
            if args.mode == 'mcm':
                auroc = 0
                fpr = 100
                for k, v in OODs.items():
                    logging.info("-"*10+f" OOD: {k} "+"-"*10)
                    zeroshot_classifier = in_classifier
                    auroc_, fpr_ = evaluate(model, zeroshot_classifier, in_image_loader, v, device=device, args=args)
                    auroc += auroc_
                    fpr += fpr_
                best = auroc / len(OODs)
                best_fpr = fpr / len(OODs)
            else:
                logging.info(f"Get: In-distribution features")
                if exp == 0:
                    ind_features = get_features(model, in_image_loader, device=device, args=args)
                    OODs_features = {}
                    for k, v in OODs.items():
                        logging.info(f"Get: {k} features")
                        OODs_features[k] = get_features(model, v, device=device, args=args)

        for epoch in range(args.epochs):
            logging.info('epoch %d'%epoch)
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, in_classifier, model.module.ood_classifier_, optimizer, in_text_features, ood_text_loader, device, epoch, args, ind_features, OODs_features, processor=processor)
            zeroshot_classifier = torch.cat([in_classifier, normalize(model.module.ood_classifier_, dim=0)], dim=1)
            lr_scheduler.step()
            if utils.is_main_process():
                auroc = 0
                fpr = 0
                aurocs_tmp = {}
                fprs_tmp = {}
                for k, v in OODs_features.items():
                    logging.info("-"*10+f" OOD: {k} "+"-"*10)
                    auroc_, fpr_ = evaluate_with_features(zeroshot_classifier, ind_features, v, device=device, args=args)
                    aurocs_tmp[k] = auroc_
                    fprs_tmp[k] = fpr_
                    auroc += auroc_
                    fpr += fpr_
                auroc /= len(OODs)
                fpr /= len(OODs)
                logging.info(f">>>>> Average AUROC: {auroc:.4f} <<<<<")
                logging.info(f">>>>> Average FPR: {fpr:.4f} <<<<<")
                if best < auroc:
                    best = auroc
                    best_aurocs = aurocs_tmp
                    best_fprs = fprs_tmp
        if utils.is_main_process() and args.mode == 'hftt':
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logging.info(f"Training time {total_time_str}")
            for k in OODs.keys():
                OOD_evals[k]['aurocs'].append(best_aurocs[k])
                OOD_evals[k]['fprs'].append(best_fprs[k])
        
        if args.num_exp > 1:
            del model
            del processor
            del optimizer
            del zeroshot_classifier

    if utils.is_main_process() and args.mode == 'hftt':
        logging.info("="*30)
        for k in OODs.keys():
            logging.info("-"*10+f" OOD: {k} "+"-"*10)
            if len(OOD_evals[k]['aurocs']) == 1:
                logging.info('MEAN AUROC: %.4f, STDEV: %.4f'%(OOD_evals[k]['aurocs'][0], 0))
                logging.info('MEAN FPR: %.4f, STDEV: %.4f'%(OOD_evals[k]['fprs'][0], 0))
            else:
                logging.info('MEAN AUROC: %.4f, STDEV: %.4f'%(statistics.mean(OOD_evals[k]['aurocs']), statistics.stdev(OOD_evals[k]['aurocs'])))
                logging.info('MEAN FPR: %.4f, STDEV: %.4f'%(statistics.mean(OOD_evals[k]['fprs']), statistics.stdev(OOD_evals[k]['fprs'])))

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    ''' main '''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default='hftt')
    parser.add_argument("--num-ood-classes", type=int, default=2000)
    parser.add_argument("--model", type=str, default='clip-base')
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--focal", type=float, default=1.0)
    parser.add_argument("--num-eval-in-an-epoch", type=int, default=10)
    parser.add_argument("--ind", default="imagenet", type=str, help="path to ImageNet directory")
    parser.add_argument("--ood", default="OOD", type=str, help="path to a directory that contains all OOD dataset directories")
    parser.add_argument("--ood-text-path", default="./words_alpha.txt", type=str, help="word set")
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=1, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--num-exp", type=int, default=5)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--lr", default=1.0, type=float, help="initial learning rate")
    ''''''

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--test-batch-size", default=100, type=int)
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.logit_coeff = 1. / args.temperature
    main(args)
