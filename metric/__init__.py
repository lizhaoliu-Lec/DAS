import copy

import faiss
import numpy as np
import torch
from sklearn.preprocessing import normalize
from scipy.stats import mode
from tqdm import tqdm

from metric import c_recall, c_nmi, c_f1, c_mAP_c, c_mAP_1000, c_mAP_lim
from metric import dists, rho_spectrum
from metric import e_recall, nmi, f1, mAP, mAP_c, mAP_1000, mAP_lim


def select(metric_name, opt):
    # Metrics based on euclidean distances
    if 'e_recall' in metric_name:
        k = int(metric_name.split('@')[-1])
        return e_recall.Metric(k)
    elif metric_name == 'nmi':
        return nmi.Metric()
    elif metric_name == 'mAP':
        return mAP.Metric()
    elif metric_name == 'mAP_c':
        return mAP_c.Metric()
    elif metric_name == 'mAP_lim':
        return mAP_lim.Metric()
    elif metric_name == 'mAP_1000':
        return mAP_1000.Metric()
    elif metric_name == 'f1':
        return f1.Metric()

    # Metrics based on cosine similarity
    elif 'c_recall' in metric_name:
        k = int(metric_name.split('@')[-1])
        return c_recall.Metric(k)
    elif metric_name == 'c_nmi':
        return c_nmi.Metric()
    elif metric_name == 'c_mAP_c':
        return c_mAP_c.Metric()
    elif metric_name == 'c_mAP_lim':
        return c_mAP_lim.Metric()
    elif metric_name == 'c_mAP_1000':
        return c_mAP_1000.Metric()
    elif metric_name == 'c_f1':
        return c_f1.Metric()

    # Generic Embedding space metric
    elif 'dists' in metric_name:
        mode = metric_name.split('@')[-1]
        return dists.Metric(mode)
    elif 'rho_spectrum' in metric_name:
        mode = int(metric_name.split('@')[-1])
        embed_dim = opt.rho_spectrum_embed_dim
        return rho_spectrum.Metric(embed_dim, mode=mode, opt=opt)
    else:
        raise NotImplementedError("Metric {} not available!".format(metric_name))


class MetricComputer:
    def __init__(self, metric_names, opt):
        self.pars = opt
        self.metric_names = metric_names
        self.list_of_metrics = [select(metric_name, opt) for metric_name in metric_names]
        self.requires = [metric.requires for metric in self.list_of_metrics]
        self.requires = list(set([x for y in self.requires for x in y]))

    def compute_standard(self, opt, model, dataloader, eval_types, device):
        eval_types = copy.deepcopy(eval_types)

        n_classes = opt.n_classes
        # image_paths = np.array([x[0] for x in dataloader.dataset.image_list])
        _ = model.eval()

        feature_colls = {key: [] for key in eval_types}

        with torch.no_grad():
            target_labels = []
            final_iter = tqdm(dataloader, desc='Embedding Data...'.format(len(eval_types)))
            # image_paths = [x[0] for x in dataloader.dataset.image_list]
            for idx, inp in enumerate(final_iter):
                input_img, target = inp[1], inp[0]
                target_labels.extend(target.numpy().tolist())
                out = model(input_img.to(device))
                if isinstance(out, tuple):
                    out, aux_f = out

                # Include embeddings of all output features
                for eval_type in eval_types:
                    if isinstance(out, dict):
                        feature_colls[eval_type].extend(out[eval_type].cpu().detach().numpy().tolist())
                    else:
                        feature_colls[eval_type].extend(out.cpu().detach().numpy().tolist())

            target_labels = np.hstack(target_labels).reshape(-1, 1)

        computed_metrics = {eval_type: {} for eval_type in eval_types}
        extra_infos = {eval_type: {} for eval_type in eval_types}

        faiss.omp_set_num_threads(self.pars.kernels)
        res = None
        torch.cuda.empty_cache()
        if self.pars.evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        for eval_type in eval_types:
            features = np.vstack(feature_colls[eval_type]).astype('float32')
            features_cosine = normalize(features, axis=1)

            """============ Compute k-Means ==============="""
            if 'kmeans' in self.requires:
                # Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features.shape[-1])
                if res is not None:
                    cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans = faiss.Clustering(features.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                # Train Kmeans
                kmeans.train(features, cluster_idx)
                centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])

            if 'kmeans_cosine' in self.requires:
                # Set CPU Cluster index
                # TODO check if IndexFlatIP is a right func
                cluster_idx = faiss.IndexFlatL2(features_cosine.shape[-1])
                if res is not None:
                    cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans = faiss.Clustering(features_cosine.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                # Train Kmeans
                kmeans.train(features_cosine, cluster_idx)
                centroids_cosine = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes,
                                                                                         features_cosine.shape[-1])
                centroids_cosine = normalize(centroids_cosine, axis=1)

            """============ Compute Cluster Labels ==============="""
            if 'kmeans_nearest' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids)
                _, computed_cluster_labels = faiss_search_index.search(features, 1)

            if 'kmeans_nearest_cosine' in self.requires:
                faiss_search_index = faiss.IndexFlatIP(centroids_cosine.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids_cosine)
                _, computed_cluster_labels_cosine = faiss_search_index.search(features_cosine, 1)

            """============ Compute Nearest Neighbours ==============="""
            if 'nearest_features' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(features.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(features)
                max_kval = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                # Return: scores, idxes
                _, k_closest_points = faiss_search_index.search(features,
                                                                int(max_kval + 1))
                k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]

            if 'nearest_features_cosine' in self.requires:
                faiss_search_index = faiss.IndexFlatIP(features_cosine.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(normalize(features_cosine, axis=1))

                max_kval = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                _, k_closest_points_cosine = faiss_search_index.search(normalize(features_cosine, axis=1),
                                                                       int(max_kval + 1))
                k_closest_classes_cosine = target_labels.reshape(-1)[k_closest_points_cosine[:, 1:]]

            if self.pars.evaluate_on_gpu:
                features = torch.from_numpy(features).to(self.pars.device)
                features_cosine = torch.from_numpy(features_cosine).to(self.pars.device)

            for metric in self.list_of_metrics:
                input_dict = {}
                if 'features' in metric.requires:
                    input_dict['features'] = features
                if 'target_labels' in metric.requires:
                    input_dict['target_labels'] = target_labels

                if 'kmeans' in metric.requires:
                    input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:
                    input_dict['computed_cluster_labels'] = computed_cluster_labels
                if 'nearest_features' in metric.requires:
                    input_dict['k_closest_classes'] = k_closest_classes

                if 'features_cosine' in metric.requires:
                    input_dict['features_cosine'] = features_cosine

                if 'kmeans_cosine' in metric.requires:
                    input_dict['centroids_cosine'] = centroids_cosine
                if 'kmeans_nearest_cosine' in metric.requires:
                    input_dict['computed_cluster_labels_cosine'] = computed_cluster_labels_cosine
                if 'nearest_features_cosine' in metric.requires:
                    input_dict['k_closest_classes_cosine'] = k_closest_classes_cosine

                computed_metrics[eval_type][metric.name] = metric(**input_dict)

            extra_infos[eval_type] = {'features': features, 'target_labels': target_labels,
                                      'image_paths': dataloader.dataset.image_paths,
                                      'query_image_paths': None, 'gallery_image_paths': None}

        torch.cuda.empty_cache()
        return computed_metrics, extra_infos
