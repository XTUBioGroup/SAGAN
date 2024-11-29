import argparse
from torch import device
import datetime
from IOStream import IOStream, table_printer, exp_init
from models import *
from train_eval import *
from data import *

parser = argparse.ArgumentParser(description='SubDR')
parser.add_argument('--data_name', default='Fdataset', help='dataset name')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--hop', type=int, default=2, help='the number of neighbor (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size during training')
parser.add_argument('--dropout_n', type=float, default=0.4, help='random drops neural node with this prob')
parser.add_argument('--dropout_e', type=float, default=0.1, help='random drops edge with this prob')
parser.add_argument('--valid_interval', type=int, default=1)
parser.add_argument('--force-undirected', action='store_true', default=False)
parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (default: 0)')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

print(args)


if __name__ == '__main__':
    seeds = [34, 42, 43, 61, 70, 83, 1024, 2014, 2047]
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m-%d_%H")
    name = args.data_name + "正常"
    path = exp_init(name, formatted_time)
    IO = IOStream(os.path.join(path, 'run.log'))
    IO.cprint(str(table_printer(args)))

    auc_lists, aupr_lists, f1_score_lists = [], [], []
    max_auroc = 0
    for seed in seeds:
        print("================================ seed=", str(seed), "===========================================")
        IO.cprint('------------------------------------ seed={}------------------------------------'.format(str(seed)))
        split_data_dict, drug_embedding, disease_embedding = load_k_fold(args.data_name, seed)

        drug_edge = build_similarity_graph(drug_embedding)
        disease_edge = build_similarity_graph(disease_embedding)
        drug_embedding = torch.tensor(drug_embedding, dtype=torch.float32)
        disease_embedding = torch.tensor(disease_embedding, dtype=torch.float32)

        for k in range(0, 10):
            print('------------------------------------ fold', str(k + 1), '------------------------------------')
            IO.cprint('------------------------------------------------- fold={}-------------------------------------------------'.format(str(k + 1)))
            train_graphs, test_graphs = extract_subgraph(split_data_dict[k], k, args)

            sim_data = SData(drug_embedding, disease_embedding, drug_edge, disease_edge)
            model = SAGAN(
                train_graphs,
                latent_dim=[128, 64, 1],
                dropout_n=args.dropout_n,
                dropout_e=args.dropout_e,
                heads=[6, 6, 6],
                force_undirected=args.force_undirected,
                sim_data=SData(drug_embedding, disease_embedding, drug_edge, disease_edge)
            )

            print('Used #train graphs: %d, #test graphs: %d' % (
                len(train_graphs),
                len(test_graphs),
            ))

            auroc, aupr, f1_score, _, _ = train_epochs(train_graphs, test_graphs, model, args, IO, sim_data)
            auc_lists.append(auroc)
            aupr_lists.append(aupr)
            f1_score_lists.append(f1_score)

            if auroc > max_auroc:
                max_auroc = auroc
                model_save_path = os.path.join(path, 'best_model.pth')
                torch.save(model, model_save_path)

    print("auroc_list", auc_lists)
    print("aupr_list", aupr_lists)
    print("f1_score_list", f1_score_lists)
    print("average auc", np.mean(auc_lists), "average aupr", np.mean(aupr_lists), "average f1_score",
          np.mean(f1_score_lists))

    IO.cprint("auroc_list {}\naupr_list {}\nf1_score_list {}\n"
              "average auc {:.4f} average aupr {:.4f} average f1_score {:.4f}".format(
        auc_lists, aupr_lists, f1_score_lists,
        np.mean(auc_lists), np.mean(aupr_lists), np.mean(f1_score_lists)
    ))
