import math
import torch.distributed as dist
from torch.utils.data import Sampler
import os
import torch
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def make_dataloader(dataset, batch_size, num_workers, world_size=None, rank=None, train=True):
    """ Create (disributed) dataloader """

    if world_size is not None and world_size > 1:
        if train:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)

        parallel_batch_size = int(batch_size/world_size)
        dataloader = DataLoader(dataset, batch_size=parallel_batch_size, shuffle=(sampler is None),
                                sampler=sampler, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

    return dataloader


def save_model(model, dir, id, gpu=""):
    """ Save a model """
    os.makedirs(dir, exist_ok=True)
    if gpu != "":
        gpu = "_" + str(gpu)

    torch.save(model.state_dict(), os.path.join(dir, id + gpu + ".pt"))


def load_model(model, dir, id, gpu=""):
    """ Load a state dict into a model """
    if gpu != "":
        gpu = "_" + str(gpu)
    state_dict = torch.load(os.path.join(dir, id + gpu + ".pt"))
    model.load_state_dict(state_dict)
    return model


class DistributedEvalSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
