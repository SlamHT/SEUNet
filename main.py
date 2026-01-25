import torch
import argparse
import os
import random
import numpy as np
import torch.multiprocessing as mp
from e3nn.o3 import Irreps, spherical_harmonics
from models.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from ppis.train import train
from models.seunet.seunet import SEUNET


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay')
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved_models",
                        help='Directory in which to save models')
    parser.add_argument('--dataset', type=str, default="ppis",
                        help='Data set')
    parser.add_argument('--trainset', type=str, default="Train_335",
                        help='Training set')
    parser.add_argument('--testset', type=str, default="Test_60",
                        help='Test set')
    parser.add_argument('--root', type=str, default="datasets",
                        help='Data set location')
    parser.add_argument('--radius', type=float, default=2,
                        help='Radius (Angstrom) between which atoms to add links.')
    parser.add_argument('--Lambda', type=float, default=0.1,
                        help='The weight of connection loss.')
    parser.add_argument('--dropoutRate', type=float, default=0.05,
                        help='dropout rate to set.')
    parser.add_argument('--K', type=float, default=35,
                        help='K.')
    parser.add_argument('--w', type=float, default=1/3,
                        help='w.')
    parser.add_argument('--model', type=str, default="seunet",
                        help='Model name')
    parser.add_argument('--hidden_features', type=int, default=128,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_h', type=int, default=2,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_attr', type=int, default=3,
                        help='max degree of geometric attribute embedding')
    parser.add_argument('--subspace_type', type=str, default="weightbalanced",
                        help='How to divide spherical harmonic subspaces')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of message passing layers')
    parser.add_argument('--high_layers', type=int, default=3,
                        help='Number of high message passing layers')
    parser.add_argument('--norm', type=str, default="instance",
                        help='Normalisation type [instance, batch]')
    parser.add_argument('--pool', type=str, default="avg",
                        help='Pooling type type [avg, sum]')

    args = parser.parse_args()
    
    input_irreps = Irreps("1090x0e")
    output_irreps = Irreps("1x0e")
    edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
    node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
    additional_message_irreps = Irreps("1x0e")

    # Create hidden irreps
    if args.subspace_type == "weightbalanced":
        hidden_irreps = WeightBalancedIrreps(
            Irreps("{}x0e".format(args.hidden_features)), node_attr_irreps, sh=True, lmax=args.lmax_h)
    elif args.subspace_type == "balanced":
        hidden_irreps = BalancedIrreps(args.lmax_h, args.hidden_features, True)
    else:
        raise Exception("Subspace type not found")
        
    first_decoder_irreps = Irreps("{}x0e".format(args.hidden_features * 2))
    print('K:' + str(args.K))
    print('w:' + str(args.w))
    model = SEUNET(args.lmax_attr,
                      input_irreps,
                      hidden_irreps,
                      first_decoder_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      num_layers=args.layers,
                      num_high_layers=args.high_layers,
                      norm=args.norm,
                      pool=args.pool,
                      dropout_rate=args.dropoutRate,
                      K=args.K,
                      additional_message_irreps=additional_message_irreps)
    args.ID = "_".join([args.model, args.dataset, str(np.random.randint(1e4, 1e5))])

    print(model)
    print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))
    print('Starting training on a single gpu...')
    args.mode = 'gpu'
    train(0, model, args)
