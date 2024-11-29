import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

def train_epochs(train_dataset, test_dataset, model, args, IO, sim_data):
    num_workers = 2
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)

    test_size = 1024
    test_loader = DataLoader(test_dataset, test_size, shuffle=False, num_workers=num_workers)

    model.to(args.device).reset_parameters()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint('Model Parameter: {}'.format(total_params))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    start_epoch = 1
    pbar = range(start_epoch, args.epochs + start_epoch)
    best_epoch, best_auc, best_aupr = 0, 0, 0
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, args.device, sim_data)
        if epoch % args.valid_interval == 0:
            train_roc_auc, train_aupr, f1_score, train_scores, train_labels = evaluate_metric(model, train_loader, args.device, epoch, sim_data)
            test_roc_auc, test_aupr, f1_score, test_scores, test_labels = evaluate_metric(model, test_loader, args.device, epoch, sim_data)
            print("epoch {}".format(epoch),
                  "train_loss {0:.4f}".format(train_loss),
                  "train_roc_auc {0:.4f}".format(train_roc_auc), "train_aupr {0:.4f}".format(train_aupr),
                  "test_roc_auc {0:.4f}".format(test_roc_auc), "test_aupr {0:.4f}".format(test_aupr))

            IO.cprint(
                "epoch {} train_loss {:.4f} train_roc_auc {:.4f} train_aupr {:.4f} test_roc_auc {:.4f} test_aupr {:.4f}".format(
                    epoch,
                    train_loss,
                    train_roc_auc,
                    train_aupr,
                    test_roc_auc,
                    test_aupr
                ))

            if test_aupr > best_aupr:
                best_epoch = epoch
                best_auc, best_aupr, best_f1_score = test_roc_auc, test_aupr, f1_score
                best_scores, best_labels = test_scores, test_labels

    print("best_epoch {}".format(best_epoch), "best_auc {0:.4f}".format(best_auc), "aupr {0:.4f}".format(best_aupr),
          "f1_score {0:.4f}".format(best_f1_score))
    IO.cprint("best_epoch {} best_auc {:.4f} aupr {:.4f} f1_score {:.4f}".format(
        best_epoch,
        best_auc,
        best_aupr,
        best_f1_score
    ))

    return best_auc, best_aupr, best_f1_score, best_scores, best_labels


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device, sim_data=None):
    if sim_data is not None:
        sim_data.drug_embedding = sim_data.drug_embedding.to(device)
        sim_data.disease_embedding = sim_data.disease_embedding.to(device)
        sim_data.drug_edge = tuple(element.to(device) if isinstance(element, torch.Tensor) else element for element in sim_data.drug_edge)
        sim_data.disease_edge = tuple(element.to(device) if isinstance(element, torch.Tensor) else element for element in sim_data.disease_edge)

    model.train()
    total_loss = 0
    pbar = loader
    for data in pbar:
        optimizer.zero_grad()
        data = data.to(device)
        true_label = data.y.view(-1).to(device)
        predict = model(data, sim_data)
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(predict, true_label)

        loss.backward()

        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


def evaluate_metric(model, loader, device, epoch, sim_data=None):
    model.eval()
    pbar = loader
    roc_auc, aupr = None, None
    all_outputs = []
    all_labels = []
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out = model(data, sim_data)

        y_true = data.y.view(-1).cpu().tolist()
        y_score = out.cpu().numpy().tolist()

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)

        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        aupr = metrics.auc(recall, precision)

        all_outputs.extend(y_score)
        all_labels.extend(y_true)

        threshold = 0.5
        binary_outputs = (np.array(all_outputs) >= threshold).astype(int)
        f1_score_val = metrics.f1_score(all_labels, binary_outputs)

        torch.cuda.empty_cache()
    return roc_auc, aupr, f1_score_val, all_outputs, all_labels
