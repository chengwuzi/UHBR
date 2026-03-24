import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


def Split_HyperGraph_to_device(H, device, split_num=16):
    H_list = []
    length = H.shape[0] // split_num
    for i in range(split_num):
        if i == split_num - 1:
            H_list.append(H[length * i : H.shape[0]])
        else:
            H_list.append(H[length * i : length * (i + 1)])
    H_split = [SparseTensor.from_scipy(H_i).to(device) for H_i in H_list]
    return H_split


def normalize_Hyper(H):
    D_v = sp.diags(1 / (np.sqrt(H.sum(axis=1).A.ravel()) + 1e-8))
    D_e = sp.diags(1 / (np.sqrt(H.sum(axis=0).A.ravel()) + 1e-8))
    H_nomalized = D_v @ H @ D_e @ H.T @ D_v
    return H_nomalized


def mix_hypergraph(raw_graph, threshold=10):
    ui_graph, bi_graph, ub_graph = raw_graph

    uu_graph = ub_graph @ ub_graph.T
    for i in range(ub_graph.shape[0]):
        for r in range(uu_graph.indptr[i], uu_graph.indptr[i + 1]):
            uu_graph.data[r] = 1 if uu_graph.data[r] > threshold else 0

    bb_graph = ub_graph.T @ ub_graph
    for i in range(ub_graph.shape[1]):
        for r in range(bb_graph.indptr[i], bb_graph.indptr[i + 1]):
            bb_graph.data[r] = 1 if bb_graph.data[r] > threshold else 0

    H = sp.vstack((ui_graph, bi_graph))
    non_atom_graph = sp.vstack((ub_graph, bb_graph))
    non_atom_graph = sp.hstack((non_atom_graph, sp.vstack((uu_graph, ub_graph.T))))
    H = sp.hstack((H, non_atom_graph))
    return H


class UHBR(nn.Module):
    def __init__(self, raw_graph, device, dp, l2_norm, emb_size=64):
        super().__init__()

        ui_graph, bi_graph, ub_graph = raw_graph
        self.num_users, self.num_bundles, self.num_items = (
            ub_graph.shape[0],
            ub_graph.shape[1],
            ui_graph.shape[1],
        )
        H = mix_hypergraph(raw_graph)
        self.atom_graph = Split_HyperGraph_to_device(normalize_Hyper(H), device)

        print("finish generating hypergraph")
        # embeddings
        self.users_feature = nn.Parameter(
            torch.FloatTensor(self.num_users, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.bundles_feature = nn.Parameter(
            torch.FloatTensor(self.num_bundles, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.user_bound = nn.Parameter(
            torch.FloatTensor(emb_size, 1).normal_(0, 0.5 / emb_size)
        )
        self.drop = nn.Dropout(dp)
        self.embed_L2_norm = l2_norm

    def propagate(self):
        embed_0 = torch.cat([self.users_feature, self.bundles_feature], dim=0)
        embed_1 = torch.cat([G @ embed_0 for G in self.atom_graph], dim=0)
        all_embeds = embed_0 / 2 + self.drop(embed_1) / 3
        users_feature, bundles_feature = torch.split(
            all_embeds, [self.num_users, self.num_bundles], dim=0
        )

        return users_feature, bundles_feature

    def predict(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, 2)
        return pred

    def regularize(self, users_feature, bundles_feature):
        loss = self.embed_L2_norm * (
            (users_feature ** 2).sum() + (bundles_feature ** 2).sum()
        )
        return loss

    def forward(self, users, bundles):
        users_feature, bundles_feature = self.propagate()
        users_embedding = users_feature[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_feature[bundles]
        pred = self.predict(users_embedding, bundles_embedding)
        loss = self.regularize(users_feature, bundles_feature)
        user_score_bound = users_feature[users] @ self.user_bound
        return pred, user_score_bound, loss

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature = users_feature[users]
        scores = users_feature @ (bundles_feature.T)
        return scores


def build_ubi_graph(raw_graph):
    ui_graph, bi_graph, ub_graph = raw_graph
    num_users, num_items = ui_graph.shape
    num_bundles = bi_graph.shape[0]

    # Node offset:
    # user: [0, num_users)
    # bundle: [num_users, num_users + num_bundles)
    # item: [num_users + num_bundles, num_users + num_bundles + num_items)

    O_uu = sp.csr_matrix((num_users, num_users))
    O_bb = sp.csr_matrix((num_bundles, num_bundles))
    O_ii = sp.csr_matrix((num_items, num_items))

    # A = [ 0      A_UB   A_UI ]
    #     [ A_UB^T 0      A_BI ]
    #     [ A_UI^T A_BI^T 0    ]
    row1 = sp.hstack([O_uu, ub_graph, ui_graph])
    row2 = sp.hstack([ub_graph.T, O_bb, bi_graph])
    row3 = sp.hstack([ui_graph.T, bi_graph.T, O_ii])

    A_ubi = sp.vstack([row1, row2, row3]).tocsr()

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    rowsum = A_ubi.sum(axis=1).A.ravel()
    d_inv_sqrt = 1 / (np.sqrt(rowsum) + 1e-8)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    A_ubi_norm = D_inv_sqrt @ A_ubi @ D_inv_sqrt
    return A_ubi, A_ubi_norm


def build_norm_ub_graph(ub_graph):
    # A_ub_norm = D_u^{-1/2} A_UB D_b^{-1/2}
    # D_u is degree of users (row sum)
    # D_b is degree of bundles (col sum)
    
    # Calculate D_u^{-1/2}
    rowsum = ub_graph.sum(axis=1).A.ravel()
    d_u_inv_sqrt = 1 / (np.sqrt(rowsum) + 1e-8)
    D_u_inv_sqrt = sp.diags(d_u_inv_sqrt)

    # Calculate D_b^{-1/2}
    colsum = ub_graph.sum(axis=0).A.ravel()
    d_b_inv_sqrt = 1 / (np.sqrt(colsum) + 1e-8)
    D_b_inv_sqrt = sp.diags(d_b_inv_sqrt)

    A_ub_norm = D_u_inv_sqrt @ ub_graph @ D_b_inv_sqrt
    return A_ub_norm


class UBIGraph(nn.Module):
    def __init__(self, raw_graph, device, dp, l2_norm, lambda0=0.5, lambda1=0.5, alpha_refine=1.0, beta_refine=0.5, emb_size=64):
        super().__init__()

        ui_graph, bi_graph, ub_graph = raw_graph
        self.num_users, self.num_bundles, self.num_items = (
            ub_graph.shape[0],
            ub_graph.shape[1],
            ui_graph.shape[1],
        )
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.alpha_refine = alpha_refine
        self.beta_refine = beta_refine
        
        self.embed_L2_norm = l2_norm
        self.drop = nn.Dropout(dp)

        self.A_ubi, A_ubi_norm = build_ubi_graph(raw_graph)
        self.A_ubi_norm_split = Split_HyperGraph_to_device(A_ubi_norm, device)
        
        # U-B Refinement Graph
        A_ub_refine_norm = build_norm_ub_graph(ub_graph)
        self.A_ub_refine_norm_t = SparseTensor.from_scipy(A_ub_refine_norm).to(device)
        self.A_ub_refine_norm_t_T = SparseTensor.from_scipy(A_ub_refine_norm.T).to(device)

        print("finish generating ubi unified graph")
        print(f"A_UB.shape: {ub_graph.shape}")
        print(f"A_UI.shape: {ui_graph.shape}")
        print(f"A_BI.shape: {bi_graph.shape}")
        print(f"A_ubi.shape: {self.A_ubi.shape}")
        
        # embeddings
        self.users_feature = nn.Parameter(
            torch.FloatTensor(self.num_users, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.bundles_feature = nn.Parameter(
            torch.FloatTensor(self.num_bundles, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, emb_size).normal_(0, 0.5 / emb_size)
        )
        self.user_bound = nn.Parameter(
            torch.FloatTensor(emb_size, 1).normal_(0, 0.5 / emb_size)
        )

    def propagate(self):
        # init embeddings
        eu0 = self.users_feature
        eb0 = self.bundles_feature
        ei0 = self.items_feature

        # concat
        e0 = torch.cat([eu0, eb0, ei0], dim=0)

        # one-hop propagation (split multiplication to avoid OOM)
        e1 = torch.cat([G @ e0 for G in self.A_ubi_norm_split], dim=0)

        # shallow fusion
        e_star = self.lambda0 * e0 + self.lambda1 * self.drop(e1)

        # split back
        Eu_star = e_star[:self.num_users]
        Eb_star = e_star[self.num_users:self.num_users + self.num_bundles]
        Ei_star = e_star[self.num_users + self.num_bundles:]
        
        # U-B Refinement
        Ru = self.A_ub_refine_norm_t @ Eb_star
        Rb = self.A_ub_refine_norm_t_T @ Eu_star
        
        Eu_hat = self.alpha_refine * Eu_star + self.beta_refine * Ru
        Eb_hat = self.alpha_refine * Eb_star + self.beta_refine * Rb

        if not hasattr(self, '_printed_shapes'):
            print(f"E0.shape: {e0.shape}")
            print(f"E1.shape: {e1.shape}")
            print(f"Eu*.shape: {Eu_star.shape}")
            print(f"Eb*.shape: {Eb_star.shape}")
            print(f"Ei*.shape: {Ei_star.shape}")
            print(f"A_ub_refine.shape: {self.A_ub_refine_norm_t.sizes()}")
            print(f"Ru.shape: {Ru.shape}")
            print(f"Rb.shape: {Rb.shape}")
            print(f"Eu_hat.shape: {Eu_hat.shape}")
            print(f"Eb_hat.shape: {Eb_hat.shape}")
            self._printed_shapes = True

        # 最终打分使用 Eu_hat 和 Eb_hat，返回 Ei_star 以便计算正则化
        return Eu_hat, Eb_hat, Ei_star

    def predict(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, 2)
        return pred

    def regularize(self, users_feature, bundles_feature, items_feature):
        loss = self.embed_L2_norm * (
            (users_feature ** 2).sum() + (bundles_feature ** 2).sum() + (items_feature ** 2).sum()
        )
        return loss

    def forward(self, users, bundles):
        users_feature, bundles_feature, items_feature = self.propagate()
        users_embedding = users_feature[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_feature[bundles]
        pred = self.predict(users_embedding, bundles_embedding)
        loss = self.regularize(users_feature, bundles_feature, items_feature)
        user_score_bound = users_feature[users] @ self.user_bound
        return pred, user_score_bound, loss

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature, items_feature = propagate_result
        users_feature = users_feature[users]
        scores = users_feature @ (bundles_feature.T)
        return scores
