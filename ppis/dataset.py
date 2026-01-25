from typing import Optional, Callable, List

import sys
import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
from torch_geometric.nn import radius_graph
from e3nn.o3 import Irreps, spherical_harmonics


class TargetGetter(object):
    """ Gets relevant target """

    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        # Specify target.
        return data


class ProteinDataset(InMemoryDataset):
    def __init__(self, root, target, radius, partition, lmax_attr, trainset, testset):
        assert partition in ["train", "valid", "test"]
        self.root = osp.abspath(osp.join(root, "ppis"))
        self.radius = radius
        self.partition = partition
        self.lmax_attr = lmax_attr
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.target = target

        self.partition_to_file = {'train': trainset + '.pkl', 'valid': testset + '.pkl', 'test': testset + '.pkl'}
        pkl_file = open('./raw/Feature/psepos/' + trainset + '_psepos_SC.pkl', 'rb')
        train_psepos_list = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open('./raw/Feature/psepos/' + testset + '_psepos_SC.pkl', 'rb')
        test_psepos_list = pickle.load(pkl_file)
        pkl_file.close()
        self.residue_psepos_dict = {**train_psepos_list, **test_psepos_list}
        
        transform = TargetGetter(self.target)

        super().__init__(self.root, transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def calc_stats(self):
        whole_y_list = []
        for data in self:
            whole_y_list.extend(data.y)
        ys = np.array(data.y).astype(int)
        mean = np.mean(ys)
        mad = np.mean(np.abs(ys - mean))
        return mean, mad

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['take_space.txt']
        except ImportError:
            print("Please install rdkit")
            return

    @property
    def processed_file_names(self) -> str:
        return ["_".join([self.partition, "r=" + str(np.round(self.radius, 2)), "l=" + str(self.lmax_attr)]) + '.pt']

    def download(self):
        try:
            import rdkit  # noqa
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            print("Please install rdkit")
            return

        print("Processing", self.partition, "with radius=" + str(np.round(self.radius, 2)) +
              ",", "l_attr=" + str(self.lmax_attr), "and", "features.")

        path = './raw/Dataset/'

        file = open(path + self.partition_to_file[self.partition], 'rb')
        partition_dict = pickle.load(file)
        file.close()
        feature_path = './raw/Feature/'
        data_list = []
        cnt = 0
        for protein in partition_dict:
            if protein == '2j3rA':
                continue
            label = torch.tensor(partition_dict[protein][-1], dtype=int)
            pssm = np.load(feature_path + 'pssm/' + protein + '.npy')
            resAF = np.load(feature_path + 'resAF/' + protein + '.npy')
            dssp = np.load(feature_path + 'dssp/' + protein + '.npy')
            hmm = np.load(feature_path + 'hmm/' + protein + '.npy')
            protein_len = np.load(feature_path + 'len/' + protein + '.npy')
            HSEexposureCA = np.load(feature_path + 'exposureCA_4_sin_cos/' + protein + '.npy')
            protbert_embeddings_arr = np.load(feature_path + 'ft_protbert/' + protein + '.npy')
            try:
                pos = self.residue_psepos_dict[protein]
            except Exception as err:
                continue

            pos = torch.from_numpy(pos)
            edge_index = radius_graph(pos, r=self.radius, loop=False)
            
            local_edge_index = torch.from_numpy(np.load(feature_path + 'local_edge_index_contact/' + protein + '.npy'))
            local_edge_index = local_edge_index.long()
            local_rel_pos = pos[local_edge_index[0]] - pos[local_edge_index[1]]  # pos_j - pos_i (note in edge_index stores tuples like (j,i))
            local_edge_dist2 = torch.sqrt(local_rel_pos.pow(2).sum(-1, keepdims=True))

            edge_index = edge_index.long()
            rel_pos = pos[edge_index[0]] - pos[edge_index[1]]  # pos_j - pos_i (note in edge_index stores tuples like (j,i))
            edge_dist2 = torch.sqrt(rel_pos.pow(2).sum(-1, keepdims=True))
            protbert_embeddings_arr = (protbert_embeddings_arr - protbert_embeddings_arr.min()) / (protbert_embeddings_arr.max() - protbert_embeddings_arr.min())
            try:
                x = torch.from_numpy(np.concatenate([pssm, hmm, dssp, resAF, HSEexposureCA, protein_len, protbert_embeddings_arr], -1).astype('float32'))
            except Exception as err:
                continue

            pos = pos.float()

            cossim = torch.nn.CosineSimilarity(dim=1)
            protein_cos = cossim(pos[edge_index[0]], pos[edge_index[1]])
            protein_cos = protein_cos.unsqueeze(-1)

            local_protein_cos = cossim(pos[local_edge_index[0]], pos[local_edge_index[1]])
            local_protein_cos = local_protein_cos.unsqueeze(-1)

            local_edge_attr, local_node_attr, local_edge_dist = self.get_O3_attr(local_edge_index, pos, self.attr_irreps, local_edge_dist2, local_protein_cos)
            
            edge_attr, node_attr, edge_dist = self.get_O3_attr(edge_index, pos, self.attr_irreps, edge_dist2, protein_cos)
            index = cnt
            name = cnt
            if x.shape[0] > node_attr.shape[0]:
                print('continue')
                continue
            data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr, additional_message_features=edge_dist, y=label, name=name, index=index, local_edge_index=local_edge_index, local_edge_attr=local_edge_attr, local_node_attr=local_node_attr, local_additional_message_features=local_edge_dist)
            data_list.append(data)
            cnt += 1
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('cnt:' + str(cnt))

    def get_O3_attr(self, edge_index, pos, attr_irreps, edge_dist2, protein_cos):
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        edge_dist2 = edge_dist2.float()
        edge_attr = spherical_harmonics(attr_irreps, rel_pos, normalize=True, normalization='component')
        node_attr = scatter(edge_attr, edge_index[1], dim=0, reduce="mean")
        return edge_attr, node_attr, edge_dist2


if __name__ == "__main__":
    dataset = ProteinDataset("datasets", 'node', 14.0, "train", lmax_attr=3)

