import torch
import torch.distributed as dist
import numpy as np

from evaluation_supp import compute_roc, compute_aupr, compute_mcc, micro_score, acc_score, compute_performance


def evaluate(model, dataloader, criterion, device, world_size, Lambda, loc=0, scale=1, epoch=0):
    """ Evaluate a model on a specific dataloader, with distributed communication (if necessary) """
    model.eval()
    N = torch.zeros(1).to(device)
    score = torch.zeros(1).to(device)
    score_bse = torch.zeros(1).to(device)
    score_connection_loss = torch.zeros(1).to(device)

    all_trues = []
    all_preds = []

    with torch.no_grad():
        for step, graph in enumerate(dataloader):
            graph = graph.to(device)
            out = model(graph, epoch).squeeze()
            for_weights = graph.y.float()
            weight_var = for_weights + 1/3
            score_bse += torch.nn.functional.binary_cross_entropy(out, graph.y.float(), weight_var).cuda()
            score_connection_loss += Lambda * model.cut_loss

            n = graph.y.size(0)
            N += n
            score += n*criterion(out*scale + loc, graph.y)
            true_label = graph.y
            all_trues.append(true_label.cpu().numpy())
            all_preds.append(out.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    print('binary_cross_entropy_loss:' + str(score_bse) + " connection_loss:" + str(score_connection_loss))

    try:
        auc = compute_roc(all_preds, all_trues)
        aupr = compute_aupr(all_preds, all_trues)
        f_max, p_max, r_max, t_max, predictions_max, threshold_max = compute_performance(all_preds, all_trues)
        acc_val = acc_score(predictions_max, all_trues)
        mcc = compute_mcc(predictions_max, all_trues)
    
        print((acc_val, f_max, p_max, r_max, auc, aupr, t_max, mcc, threshold_max))
    except Exception as e:
        print(e)
        return batch_time, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    model.train()
    if world_size > 1:
        dist.all_reduce(score)
        dist.all_reduce(N)

    return f_max.item()
