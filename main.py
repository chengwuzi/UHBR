import argparse
from time import time

import torch
from torch.utils.data import DataLoader
import dataset
from model import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Go UBI-graph for bundle recommendation")
    parser.add_argument("--lr", type=float, default=5e-3, help="the learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Youshu",
        help="available datasets: [Youshu, NetEase]",
    )
    parser.add_argument("--epochs", type=int, default=100, help="the number of epochs")
    parser.add_argument("--dp", type=float, default=0.2, help="the dropout rate")
    parser.add_argument("--alpha", type=int, default=8, help="alpha in UIBloss")
    parser.add_argument("--l2_norm", type=float, default=0.1, help="l2 norm")
    parser.add_argument("--model_type", type=str, default="ubi_graph", help="model type: [UHBR, ubi_graph]")
    parser.add_argument("--lambda0", type=float, default=0.05, help="shallow fusion weight 0")
    parser.add_argument("--lambda1", type=float, default=0.95, help="shallow fusion weight 1")
    parser.add_argument("--alpha_refine", type=float, default=1.0, help="UB refinement self weight")
    parser.add_argument("--beta_refine", type=float, default=0.0, help="UB refinement neigh weight")
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


def print_summary(history):
    if not history:
        print("No evaluation results to summarize.")
        return

    # 按照 score 降序排序
    history.sort(key=lambda x: x["score"], reverse=True)
    
    best_epoch = history[0]
    
    top_k = min(3, len(history))
    top_epochs = history[:top_k]
    
    avg_recall20 = sum(x["recall20"] for x in top_epochs) / top_k
    avg_recall40 = sum(x["recall40"] for x in top_epochs) / top_k
    avg_recall80 = sum(x["recall80"] for x in top_epochs) / top_k
    avg_ndcg20 = sum(x["ndcg20"] for x in top_epochs) / top_k
    avg_ndcg40 = sum(x["ndcg40"] for x in top_epochs) / top_k
    avg_ndcg80 = sum(x["ndcg80"] for x in top_epochs) / top_k
    avg_score = sum(x["score"] for x in top_epochs) / top_k

    print("\n" + "="*50)
    print("FINAL TRAINING SUMMARY")
    print("="*50)
    
    print("\n[Best Epoch]")
    print(f"Epoch: {best_epoch['epoch']}")
    print(f"Recall@20: {best_epoch['recall20']:.6f}")
    print(f"Recall@40: {best_epoch['recall40']:.6f}")
    print(f"Recall@80: {best_epoch['recall80']:.6f}")
    print(f"NDCG@20:   {best_epoch['ndcg20']:.6f}")
    print(f"NDCG@40:   {best_epoch['ndcg40']:.6f}")
    print(f"NDCG@80:   {best_epoch['ndcg80']:.6f}")
    print(f"Score:     {best_epoch['score']:.6f}")
    
    print(f"\n[Top-{top_k} Epochs]")
    for i, res in enumerate(top_epochs, 1):
        print(f"Rank {i} -> Epoch: {res['epoch']:03d} | R@20: {res['recall20']:.6f} | R@40: {res['recall40']:.6f} | R@80: {res['recall80']:.6f} | N@20: {res['ndcg20']:.6f} | N@40: {res['ndcg40']:.6f} | N@80: {res['ndcg80']:.6f} | Score: {res['score']:.6f}")
        
    print(f"\n[Top-{top_k} Average]")
    print(f"Avg Recall@20: {avg_recall20:.6f}")
    print(f"Avg Recall@40: {avg_recall40:.6f}")
    print(f"Avg Recall@80: {avg_recall80:.6f}")
    print(f"Avg NDCG@20:   {avg_ndcg20:.6f}")
    print(f"Avg NDCG@40:   {avg_ndcg40:.6f}")
    print(f"Avg NDCG@80:   {avg_ndcg80:.6f}")
    print(f"Avg Score:     {avg_score:.6f}")
    print("="*50 + "\n")


def main():
    args = parse_args()
    device = torch.device("cuda")
    set_seed(123)
    (
        bundle_train_data,
        bundle_test_data,
        item_data,
        assist_data,
    ) = dataset.get_dataset(args.dataset, path="./data")
    if args.dataset == "Youshu":
        batch_size = 1024
    else:
        batch_size = 2048
    train_loader = DataLoader(
        bundle_train_data, batch_size, True, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        bundle_test_data, 4096, False, num_workers=16, pin_memory=True
    )

    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    metrics = [
        Recall(20),
        NDCG(20),
        Recall(40),
        NDCG(40),
        Recall(80),
        NDCG(80),
    ]
    loss_func = UIBLoss(alpha=args.alpha)
    graph = [ui_graph, bi_graph, ub_graph]
    
    if args.model_type == "ubi_graph":
        model = UBIGraph(
            graph, device, args.dp, args.l2_norm, 
            args.lambda0, args.lambda1,
            args.alpha_refine, args.beta_refine
        ).to(device)
    else:
        model = UHBR(graph, device, args.dp, args.l2_norm).to(device)
        
    print("num parameters")
    print(sum(p.numel() for p in model.parameters()))
    # op
    op = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        op, milestones=[35, 55, 75], gamma=0.5
    )
    
    history = []
    
    for epoch in range(args.epochs):

        train(model, epoch + 1, train_loader, op, device, loss_func)
        test_metrics = test(model, test_loader, device, metrics)
        scheduler.step()
        
        # 提取当前 epoch 的评测结果
        # 注意: metrics 列表的顺序是 [Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80), NDCG(80)]
        recall20 = float(test_metrics[0].metric)
        ndcg20 = float(test_metrics[1].metric)
        recall40 = float(test_metrics[2].metric)
        ndcg40 = float(test_metrics[3].metric)
        recall80 = float(test_metrics[4].metric)
        ndcg80 = float(test_metrics[5].metric)
        
        score = recall20 + ndcg20
        
        history.append({
            "epoch": epoch + 1,
            "recall20": recall20,
            "recall40": recall40,
            "recall80": recall80,
            "ndcg20": ndcg20,
            "ndcg40": ndcg40,
            "ndcg80": ndcg80,
            "score": score
        })
        
    print_summary(history)

if __name__ == "__main__":
    main()

