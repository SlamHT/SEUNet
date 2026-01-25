import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from e3nn.nn import BatchNorm
import numpy as np
from e3nn.o3 import Irreps, spherical_harmonics
from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm


class SEUNET(nn.Module):
    def __init__(
        self,
        lmax_attr,
        input_irreps,
        hidden_irreps,
        first_decoder_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        num_layers,
        num_high_layers,
        norm=None,
        pool="avg",
        dropout_rate=0.05,
        K=10,
        additional_message_irreps=None,
    ):
        super().__init__()

        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

        layers = []
        for i in range(num_layers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.layers = nn.ModuleList(layers)

        decoder_layer = []
        for i in range(1):
            decoder_layer.append(
                    SEGNNLayer(
                        hidden_irreps,
                        hidden_irreps,
                        hidden_irreps,
                        edge_attr_irreps,
                        node_attr_irreps,
                        norm=norm,
                        additional_message_irreps=additional_message_irreps,
                    )
                )
        self.decoder_layers = nn.ModuleList(decoder_layer)

        self.K = int(K)

        print("self.K:" + str(self.K))

        self.local_output_K_irreps = Irreps(str(self.K) + "x0e")

        local_layers = []
        local_layers.append(SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                ))
        self.local_layers = nn.ModuleList(local_layers)

        self.mlp_for_local = O3TensorProduct(hidden_irreps, self.local_output_K_irreps, node_attr_irreps)
        self.dimension_reduction = O3TensorProduct(2 * hidden_irreps, hidden_irreps, node_attr_irreps)

        high_layers = []
        for i in range(num_high_layers):
            high_layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.high_layers = nn.ModuleList(high_layers)

        self.sigmoid = nn.Sigmoid()

        self.coefficients = []
        self.alpha = 0.3
        self.coefficient_num = 1 + num_layers + 1 + 1
        cur_coefficient = self.alpha
        for i in range(self.coefficient_num):
            if i != self.coefficient_num - 1:
                cur_coefficient = self.alpha * torch.pow(1 - self.alpha, torch.tensor(i, dtype=torch.int))
            else:
                cur_coefficient = torch.pow(1 - self.alpha, torch.tensor(i, dtype=torch.int))
            self.coefficients.append(cur_coefficient)
        coefficients_sum = 0
        for i in range(self.coefficient_num):
            coefficients_sum += self.coefficients[i]
        print('coefficients_sum:' + str(coefficients_sum))

        self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
        self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.contains_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0

    def get_cut_loss(self, A):
        A = F.normalize(A, p=2, dim=-1)
        return torch.norm(A - torch.eye(A.shape[-1]).to(A.device), p="fro", dim=[0, 1]).mean()

    def get_O3_attr(self, edge_index, pos, attr_irreps):
        """ Creates spherical harmonic edge attributes and node attributes for the SEGNN """
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        edge_attr = spherical_harmonics(attr_irreps, rel_pos, normalize=True, normalization='component')
        node_attr = scatter(edge_attr, edge_index[1], dim=0, reduce="mean")
        return edge_attr, node_attr

    def construct_edges(self, adjacency_matrix, pos, batch):
        row, col = adjacency_matrix.nonzero(as_tuple=False).t()
        val = adjacency_matrix[row, col]
        row = row.unsqueeze(-1)
        col = col.unsqueeze(-1)
        high_edge_index = torch.cat((row, col), dim=-1).T
        high_additional_message_features = val.unsqueeze(-1)
        high_batch = batch[torch.arange(adjacency_matrix.shape[-1])]
        high_edge_attr, high_node_attr = self.get_O3_attr(high_edge_index, pos, self.attr_irreps)
        return high_edge_index, high_edge_attr, high_node_attr, high_batch, high_additional_message_features

    def forward(self, graph, epoch_step):
        """SEGNN forward pass"""
        x, pos, edge_index, edge_attr, node_attr, batch, additional_message_features, local_edge_index, local_edge_attr, local_node_attr, local_additional_message_features = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
            graph.additional_message_features,
            graph.local_edge_index,
            graph.local_edge_attr,
            graph.local_node_attr,
            graph.local_additional_message_features,
        )
        
        self.catch_isolated_nodes(graph)

        vec_list = []

        # Embed
        x = self.embedding_layer(x, node_attr)
        final_h_out = self.coefficients[0] * x
        vec_list.append(x)

        # Pass messages
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )
            vec_list.append(x)

        for layer in self.local_layers:
            pooling_features = layer(x, local_edge_index, local_edge_attr, local_node_attr, batch, local_additional_message_features)
        pooling_features = self.mlp_for_local(pooling_features, local_node_attr)
        s = F.softmax(pooling_features, dim=-1)
        sT = s.transpose(-1, 0)
        X = torch.mm(sT, pos)
        H = torch.mm(sT, x)
        s_sum = torch.sum(s, dim=0)
        s_sum_K_3 = s_sum.repeat(X.shape[-1], 1).T
        s_sum_K_d = s_sum.repeat(H.shape[-1], 1).T
        X = X / (s_sum_K_3 + 1e-8)
        H = H / (s_sum_K_d + 1e-8)
        a = spmm(torch.stack((local_edge_index[0], local_edge_index[1]), dim=0),
                 torch.ones_like(local_edge_index[0]), x.shape[0], x.shape[0], s)  # [N, K]
        A = torch.mm(sT, a)
        self.cut_loss = self.get_cut_loss(A)
        row, col = edge_index
        aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), x.shape[0], x.shape[0], s)
        AA = torch.mm(sT, aa)
        high_edge_index, high_edge_attr, high_node_attr, high_batch, high_additional_message_features = self.construct_edges(AA, X, batch)  
        for layer in self.high_layers:
            l_H = layer(H, high_edge_index, high_edge_attr, high_node_attr, high_batch, high_additional_message_features)
        l_H = torch.mm(s, l_H).reshape(-1, l_H.shape[-1])

        h_out = torch.cat((x, l_H), dim=-1)
        h_out = self.dimension_reduction(h_out, node_attr)

        vec_list.append(h_out)

        for layer in self.decoder_layers:
            h_out = layer(h_out, edge_index, edge_attr, node_attr, batch, additional_message_features)
        vec_list.append(h_out)
        for i in range(1, len(vec_list)):
            final_h_out += self.coefficients[i] * vec_list[i]
        final_h_out = self.pre_pool1(final_h_out, node_attr)
        final_h_out = self.pre_pool2(final_h_out, node_attr)
        final_h_out = self.sigmoid(final_h_out)
        return final_h_out


class SEGNNLayer(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""

        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x = 0.7 * x + 0.3 * update
        return x
