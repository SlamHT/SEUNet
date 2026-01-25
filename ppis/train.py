import time
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ppis.dataset import ProteinDataset
from ppis.evaluate import evaluate
import utils


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def train(gpu, model, args):

    device = 'cuda:' + str(gpu)

    model = model.to(device)

    # Create datasets and dataloaders
    train_loader = utils.make_dataloader(ProteinDataset(args.root, "node", args.radius, "train", args.lmax_attr, trainset=args.trainset, testset=args.testset), args.batch_size, args.num_workers, 1, gpu, train=True)
    valid_loader = utils.make_dataloader(ProteinDataset(args.root, "node", args.radius, "valid", args.lmax_attr, trainset=args.trainset, testset=args.testset), args.batch_size, args.num_workers, 1, gpu, train=False)
    test_loader = utils.make_dataloader(ProteinDataset(args.root, "node", args.radius, "test", args.lmax_attr, trainset=args.trainset, testset=args.testset), args.batch_size, args.num_workers, 1, gpu, train=False)

    # Get train set statistics
    target_mean, target_mad = train_loader.dataset.calc_stats()

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    print('Lambda:' + str(args.Lambda))

    # Logging parameters
    best_valid_F1 = 0
    i = 0
    N_samples = 0
    loss_sum = 0

    # Let's start!
    if gpu == 0:
        print("Training:", args.ID)
    for epoch in range(args.epochs):
        # Training loop

        for step, graph in enumerate(train_loader):
            graph = graph.to(device)
            out = model(graph, epoch).squeeze()
            for_weights = graph.y.float()
            weight_var = for_weights + args.w
            loss = torch.nn.functional.binary_cross_entropy(out, graph.y.float(), weight_var).cuda() + args.Lambda * model.cut_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            i += 1
            N_samples += graph.y.size(0)
            loss_sum += loss

            time.sleep(0.1)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            time.sleep(0.1)

            # Report
            if i % args.print == 0:
                print("epoch:%2d  step:%4d  loss: %0.4f " %(epoch, step, loss_sum/i))

                i = 0
                N_samples = 0
                loss_sum = 0

        # Evaluate on validation set
        try:
            valid_F1 = evaluate(model, valid_loader, criterion, device, 1, args.Lambda, target_mean, target_mad, epoch)
            # Save best validation model
            if valid_F1 > best_valid_F1:
                best_valid_F1 = valid_F1
                utils.save_model(model, args.save_dir, args.ID, device)
            print("VALIDATION: epoch:%2d  step:%4d  current best F1:%0.4f" %(epoch, step, best_valid_F1))
        except:
            print('exception in valid_MAE')
            continue

    # Final evaluation on test set
    model = utils.load_model(model, args.save_dir, args.ID, device)
    test_F1 = evaluate(model, test_loader, criterion, device, 1, args.Lambda, target_mean, target_mad, args.epochs + 1)
