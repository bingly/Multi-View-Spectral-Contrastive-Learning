import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import time
import argparse
import warnings

import utils as U
from layers import *
from metrics import *
from dataprocessing import *

import yaml
import torch
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(description='DOEN-MVC')

    parser.add_argument('--db', type=str, default='MNIST_USPS',
                        choices=['3Sources', 'BBCSport', 'MNIST_USPS'], help='dataset name')
    parser.add_argument('--kernel', type=str, default='cosine',
                        choices=['cosine','polynomial','gaussian','laplacian','imq','sigmoid', 'linear'])
    parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
    parser.add_argument("--con_epochs", default=100, help='Number of epochs to training.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initializing learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
    parser.add_argument("--temperature_l", type=float, default=1.0)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='The total number of samples must be evenly divisible by batch_size.')
    parser.add_argument('--normalized', type=bool, default=False)
    parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')
    return parser


def train(con_epochs, mv_data_loader, num_views, num_samples, num_clusters, alpha, beta, batch_size, knn=5):
    acc_array = []
    nmi_array = []
    f1_array = []
    ari_array = []
    loss_values = []
    fusion_best = None

    for epoch in range(con_epochs):
        total_fusion = []
        labels_vector = []

        Model.train()

        epoch_loss = 0

        for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
            positive_adj_graphs = U.adj_graphs(sub_data_views, batch_size, knn, 'cosine')
            fused_adj_graph = U.fused_adj_graph(positive_adj_graphs, batch_size, num_views)
            adj_graph = torch.tensor(fused_adj_graph, dtype=torch.float32, device=device)

            Hs, h_consensus = Model(sub_data_views)

            loss = Model.loss(Hs, h_consensus, adj_graph, args.temperature_l, alpha, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if epoch % 10 == 0:
                total_fusion.extend(h_consensus.detach().cpu().numpy())
                labels_vector.extend(sub_labels)

        if epoch % 10 == 0:
            labels_vector = np.array(labels_vector).reshape(num_samples)
            total_fusion = np.array(total_fusion)

            clustering = KMeans(n_clusters=num_clusters, n_init='auto', random_state=23).fit(total_fusion)
            predict_labels = clustering.labels_

            acc, nmi, ari, f1 = calculate_metrics(labels_vector, predict_labels)
            acc_array.append(acc)
            nmi_array.append(nmi)
            f1_array.append(f1)
            ari_array.append(ari)

            if len(acc_array) <= 1 or acc > max(acc_array):
                fusion_best = total_fusion

            print('Epoch {}: alpha = {} beta = {} KNN = {} Batch = {} ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} F1={:.4f} loss:{:.7f}'.format(
                    epoch, alpha, beta, knn, batch_size, acc, nmi, ari, f1, epoch_loss / num_samples))

        loss_values.append(epoch_loss / num_samples)

    return acc_array, nmi_array, f1_array, ari_array, loss_values


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    config_file = f'./config/{args.db}.yaml'
    if os.path.exists(config_file):
        with open(config_file) as f:
            if hasattr(yaml, "FullLoader"):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())
        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)

    print("==========\nArgs:{}\n==========".format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mv_data = MultiviewData(args.db, device)
    input_sizes = np.zeros(mv_data.num_views, dtype=int)
    for idx in range(mv_data.num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

    U.set_seed(args.seed)
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, args.batch_size)

    Model = DOEN_MultiView(num_views, input_sizes, args.dims, args.kernel, num_clusters,
                           args.batch_size).to(device)

    optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    t = time.time()

    acc_array, nmi_array, f1_array, ari_array, loss_values = train(args.con_epochs, mv_data_loader, num_views,
                                                                   num_samples, num_clusters, args.alpha, args.beta,
                                                                   args.batch_size, args.knn)
    exec_time = time.time() - t
    print("Total time elapsed: {}s".format(exec_time))

    best_index = np.argmax(np.array(acc_array))
    acc_max = acc_array[best_index] * 100
    nmi_max = nmi_array[best_index] * 100
    f1_max = f1_array[best_index] * 100
    ari_max = ari_array[best_index] * 100
    print('MSCL,{:.2f},{:.2f},{:.2f},{:.2f},{:.6f} alpha={} ker={} KNN={} Batch={} Seed={}'.format(
            acc_max, nmi_max, ari_max, f1_max, exec_time, args.alpha, args.kernel, args.knn, args.batch_size,args.seed))
    with open('./results/res_%s.txt' % args.db, 'a+') as f:
        f.write('MSCL,{:.2f},{:.2f},{:.2f},{:.2f},{:.6f} alpha={} ker={} KNN={} Batch={} Seed={} \n'.format(
                acc_max, nmi_max, ari_max, f1_max, exec_time, args.alpha, args.kernel, args.knn, args.batch_size, args.seed))
        f.flush()
