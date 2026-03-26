import os
import argparse
from time import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_sparse import SparseTensor

# ==================== Dataset ====================
def print_statistics(X, string):
    print(">" * 10 + string + ">" * 10)
    print("Average interactions", X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print("Non-zero rows", len(unique_nonzero_row_indice) / X.shape[0])
    print("Non-zero columns", len(unique_nonzero_col_indice) / X.shape[1])
    print("Matrix density", len(nonzero_row_indice) / (X.shape[0] * X.shape[1]))

class BasicDataset(Dataset):
    def __init__(self, path, name, task, neg_sample):
        self.path = path
        self.name = name
        self.task = task
        self.neg_sample = neg_sample
        self.num_users, self.num_bundles, self.num_items = self.__load_data_size()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __load_data_size(self):
        with open(os.path.join(self.path, self.name, "{}_data_size.txt".format(self.name)), "r") as f:
            return [int(s) for s in f.readline().split("\t")][:3]

    def load_U_B_interaction(self):
        with open(os.path.join(self.path, self.name, "user_bundle_{}.txt".format(self.task)), "r") as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split("\t")), f.readlines()))

    def load_U_I_interaction(self):
        with open(os.path.join(self.path, self.name, "user_item.txt"), "r") as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split("\t")), f.readlines()))

    def load_B_I_affiliation(self):
        with open(os.path.join(self.path, self.name, "bundle_item.txt"), "r") as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split("\t")), f.readlines()))

class BundleTrainDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, "train", 1)
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()
        print_statistics(self.ground_truth_u_b, "U-B statistics in train")

    def __getitem__(self, index):
        user_b, pos_bundle = self.U_B_pairs[index]
        all_bundles = [pos_bundle]
        while True:
            i = np.random.randint(self.num_bundles)
            if self.ground_truth_u_b[user_b, i] == 0 and not i in all_bundles:
                all_bundles.append(i)
                if len(all_bundles) == self.neg_sample + 1:
                    break
        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    def __len__(self):
        return len(self.U_B_pairs)

class BundleTestDataset(BasicDataset):
    def __init__(self, path, name, train_dataset, task="test"):
        super().__init__(path, name, task, None)
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()

        self.train_mask_u_b = train_dataset.ground_truth_u_b
        print_statistics(self.ground_truth_u_b, f"U-B statistics in {task}")
        self.users = torch.arange(self.num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(self.num_bundles, dtype=torch.long)
        assert self.train_mask_u_b.shape == self.ground_truth_u_b.shape

    def __getitem__(self, index):
        return (
            index,
            torch.from_numpy(self.ground_truth_u_b[index].toarray()).squeeze(),
            torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze(),
        )

    def __len__(self):
        return self.ground_truth_u_b.shape[0]

class ItemDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        self.U_I_pairs = self.load_U_I_interaction()
        indice = np.array(self.U_I_pairs, dtype=np.int32)
        values = np.ones(len(self.U_I_pairs), dtype=np.float32)
        self.ground_truth_u_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_users, self.num_items),
        ).tocsr()
        print_statistics(self.ground_truth_u_i, "U-I statistics")

class AssistDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        self.B_I_pairs = self.load_B_I_affiliation()
        indice = np.array(self.B_I_pairs, dtype=np.int32)
        values = np.ones(len(self.B_I_pairs), dtype=np.float32)
        self.ground_truth_b_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_bundles, self.num_items),
        ).tocsr()
        print_statistics(self.ground_truth_b_i, "B-I statistics")

def get_dataset(name, path="./datasets/"):
    assist_data = AssistDataset(path, name)
    print("finish loading assist data")
    item_data = ItemDataset(path, name)
    print("finish loading item data")

    bundle_train_data = BundleTrainDataset(path, name)
    print("finish loading bundle train data")
    bundle_val_data = BundleTestDataset(path, name, bundle_train_data, task="tune")
    print("finish loading bundle val data")
    bundle_test_data = BundleTestDataset(path, name, bundle_train_data, task="test")
    print("finish loading bundle test data")

    return bundle_train_data, bundle_val_data, bundle_test_data, item_data, assist_data


# ==================== Model ====================
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


# ==================== Utils ====================
class UIBLoss(nn.Module):
    def __init__(self, alpha=8, reduction="sum"):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, model_output, **kwargs):
        pred, user_bound, reg_loss = model_output
        loss_p = -torch.log(torch.sigmoid(pred[:, :1] - user_bound))
        loss_n = -torch.log(torch.sigmoid(user_bound - pred[:, 1:]))
        loss = loss_p + self.alpha * loss_n
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return loss + reg_loss

_is_hit_cache = {}

def get_is_hit(scores, ground_truth, topk):
    global _is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in _is_hit_cache and _is_hit_cache[topk]["id"] == cacheid:
        return _is_hit_cache[topk]["is_hit"]
    else:
        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long
        ).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        _is_hit_cache[topk] = {"id": cacheid, "is_hit": is_hit}
        return is_hit

class _Metric:
    def __init__(self):
        self.start()

    @property
    def metric(self):
        return self._metric

    @property
    def sum(self):
        return self._sum

    @property
    def cnt(self):
        return self._cnt

    def __call__(self, scores, ground_truth):
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError

    def start(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0

    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum / self._cnt

class Recall(_Metric):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.epison = 1e-8

    def get_title(self):
        return "Recall@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += (is_hit / (num_pos + self.epison)).sum().item()

class NDCG(_Metric):
    def DCG(self, hit, device=torch.device("cpu")):
        hit = hit / torch.log2(
            torch.arange(2, self.topk + 2, device=device, dtype=torch.float)
        )
        return hit.sum(-1)

    def IDCG(self, num_pos):
        hit = torch.zeros(self.topk, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.IDCGs = torch.empty(1 + self.topk, dtype=torch.float)
        self.IDCGs[0] = 1
        for i in range(1, self.topk + 1):
            self.IDCGs[i] = self.IDCG(i)

    def get_title(self):
        return "NDCG@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        device = scores.device
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).to(torch.long)
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos.cpu()]
        ndcg = dcg / idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += ndcg.sum().item()

class data_prefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device)
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_user, self.next_bundle = next(self.loader)
        except StopIteration:
            self.next_user = None
            self.next_bundle = None
            return
        with torch.cuda.stream(self.stream):
            self.next_user = self.next_user.to(self.device, non_blocking=True)
            self.next_bundle = self.next_bundle.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        user = self.next_user
        bundle = self.next_bundle
        self.preload()
        return user, bundle


# ==================== Train/Test ====================
def parse_args():
    parser = argparse.ArgumentParser(description="UHBR for bundle recommendation (Standard Version)")
    parser.add_argument("--lr", type=float, default=5e-3, help="the learning rate")
    parser.add_argument("--dataset", type=str, default="Youshu", help="available datasets: [Youshu, NetEase]")
    parser.add_argument("--epochs", type=int, default=120, help="the number of epochs")
    parser.add_argument("--dp", type=float, default=0.2, help="the dropout rate")
    parser.add_argument("--alpha", type=int, default=8, help="alpha in UIBloss")
    parser.add_argument("--l2_norm", type=float, default=0.1, help="l2 norm")
    return parser.parse_args()

def train(model, epoch, loader, optim, device, loss_func):
    prefetcher = data_prefetcher(loader, device)
    model.train()
    start = time()
    i = 0
    users, bundles = prefetcher.next()
    while users is not None:
        i += 1
        optim.zero_grad()
        modelout = model(users, bundles)
        loss = loss_func(modelout, batch_size=loader.batch_size)
        loss.backward()
        optim.step()
        if i % 20 == 0:
            print(
                "U-B Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    (i + 1) * loader.batch_size,
                    len(loader.dataset),
                    100.0 * (i + 1) / len(loader),
                    loss,
                )
            )
        users, bundles = prefetcher.next()
    print("Train Epoch: {}: time = {:d}s".format(epoch, int(time() - start)))
    return loss

def test(model, loader, device, metrics):
    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        rs = model.propagate()
        for users, ground_truth_u_b, train_mask_u_b in loader:
            pred_b = model.evaluate(rs, users.to(device))
            pred_b -= 1e8 * train_mask_u_b.to(device)
            for metric in metrics:
                metric(pred_b, ground_truth_u_b.to(device))
    print("Test: time={:.5f}s".format(int(time() - start)))
    for metric in metrics:
        metric.stop()
        print("{}:{}".format(metric.get_title(), metric.metric), end="\t")
    print("")
    return metrics

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def print_summary(best_record):
    if not best_record:
        print("No evaluation results to summarize.")
        return

    print("\n" + "="*50)
    print("FINAL TRAINING SUMMARY")
    print("="*50)
    
    print("\n### Best Epoch (selected by validation score)")
    print(f"best_epoch: {best_record['epoch']}")
    print(f"best_val_score: {best_record['val_score']:.6f}")
    
    print("\nvalidation metrics at best epoch:")
    print(f"  - Recall@10: {best_record['val_metrics'][0]:.6f} / @20: {best_record['val_metrics'][2]:.6f} / @40: {best_record['val_metrics'][4]:.6f} / @80: {best_record['val_metrics'][6]:.6f}")
    print(f"  - NDCG@10:   {best_record['val_metrics'][1]:.6f} / @20: {best_record['val_metrics'][3]:.6f} / @40: {best_record['val_metrics'][5]:.6f} / @80: {best_record['val_metrics'][7]:.6f}")
    
    print("\ntest metrics at best epoch:")
    print(f"  - Recall@10: {best_record['test_metrics'][0]:.6f} / @20: {best_record['test_metrics'][2]:.6f} / @40: {best_record['test_metrics'][4]:.6f} / @80: {best_record['test_metrics'][6]:.6f}")
    print(f"  - NDCG@10:   {best_record['test_metrics'][1]:.6f} / @20: {best_record['test_metrics'][3]:.6f} / @40: {best_record['test_metrics'][5]:.6f} / @80: {best_record['test_metrics'][7]:.6f}")
    
    print("="*50 + "\n")

def main():
    args = parse_args()
    device = torch.device("cuda")
    set_seed(123)
    
    (
        bundle_train_data,
        bundle_val_data,
        bundle_test_data,
        item_data,
        assist_data,
    ) = get_dataset(args.dataset, path="./datasets")
    
    if args.dataset == "Youshu":
        batch_size = 1024
    else:
        batch_size = 2048
        
    train_loader = DataLoader(
        bundle_train_data, batch_size, True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        bundle_val_data, 4096, False, num_workers=16, pin_memory=True
    )
    test_loader = DataLoader(
        bundle_test_data, 4096, False, num_workers=16, pin_memory=True
    )

    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    metrics = [
        Recall(10), NDCG(10),
        Recall(20), NDCG(20),
        Recall(40), NDCG(40),
        Recall(80), NDCG(80),
    ]
    loss_func = UIBLoss(alpha=args.alpha)
    graph = [ui_graph, bi_graph, ub_graph]
    
    model = UHBR(graph, device, args.dp, args.l2_norm).to(device)
        
    print("num parameters")
    print(sum(p.numel() for p in model.parameters()))
    
    op = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        op, milestones=[35, 55, 75], gamma=0.5
    )
    
    best_record = None
    best_val_score = -1.0
    
    for epoch in range(args.epochs):
        train(model, epoch + 1, train_loader, op, device, loss_func)
        
        print(f"--- Epoch {epoch + 1} Validation ---")
        val_metrics_obj = test(model, val_loader, device, metrics)
        val_metrics_vals = [float(m.metric) for m in val_metrics_obj]
        
        print(f"--- Epoch {epoch + 1} Test ---")
        test_metrics_obj = test(model, test_loader, device, metrics)
        test_metrics_vals = [float(m.metric) for m in test_metrics_obj]
        
        scheduler.step()
        
        val_score = val_metrics_vals[2] + val_metrics_vals[3]  # Recall@20 + NDCG@20
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_record = {
                "epoch": epoch + 1,
                "val_score": val_score,
                "val_metrics": val_metrics_vals,
                "test_metrics": test_metrics_vals
            }
            print(f">>> New best epoch {epoch + 1} found! Val Score: {val_score:.6f}")
        
    print_summary(best_record)

if __name__ == "__main__":
    main()
