import random

import torch
from sklearn.cluster import KMeans as skKMeans
from tqdm import tqdm

from maesy.dataset.clustering_methods.base_clustering import BaseClustering
from maesy.dataset.clustering_methods.pca import pca_reduction
from maesy.evaluation.inferer import Inferer
from maesy.model import ResnetFeatureExtractor, BaseModel


class ResnetKmeans(BaseClustering):

    def _create_model(self, forward_scale):
        # Setup model for feature extraction
        model: BaseModel = ResnetFeatureExtractor("resnet50", img_size=forward_scale, out_layers=["c6"])
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def _cluster(self, new_dataloader, existing_dataloader, model, **kwargs):
        """
        Clusters images from the given paths into n_c clusters using features extracted by a ResNet50 model and KMeans clustering.
        Args:
        :param paths: List of paths to image folders.
        :param n_c: Number of clusters to form.
        :param batch_size: Batch size for feature extraction (default: 256).
        :param forward_scale: Scale to resize images for feature extraction (default: 224).
        :param step: Step size for sampling images from folders (default: 1, i.e. use every image).
        :param start_index: Start index for sampling images from folders (default: 0).
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        n_c = kwargs.get("num_clusters", -1)
        assert n_c != -1, "Error: num_clusters must be provided as a keyword argument when using KMeans clustering"

        inferer = Inferer(model=model, data_loader=new_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        all_predictions, _ = inferer.infer()
        output_cat = torch.cat(all_predictions).to("cpu").detach().squeeze()

        out_pca = pca_reduction(output_cat, 256)
        # q = min(256, output_cat.shape[0]-1)
        # if q < 256:
        #     print(f"Warning: Not enough samples for accurate PCA reduction. Reducing to {q} dimensions instead of 256.")
        #
        # U, S, V = torch.pca_lowrank(output_cat, center=True, q=q)
        # out_pca = matmul(output_cat, V)

        # TODO: Incorporate existing_dataloader

        print("Clustering")
        start.record()
        # batch_size = 1 # batching KMeans is just like running it multiple times on different data
        # assert out_pca.shape[0]%batch_size == 0 , f"Error, {out_pca.shape[0]} is not divisible by {batch_size}"
        sk_result = skKMeans(n_clusters=n_c).fit(out_pca)
        # cluster_ids_x, cluster_centers = kmeans(X=out_pca.detach(), num_clusters=n_C, distance='euclidean', device=torch.device('cuda:0'), tol=1e-4)
        end.record()
        # torch.cuda.synchronize()
        print(f"KMeans clustering done in {start.elapsed_time(end)/1000:.2f} seconds.")

        labels = list(sk_result.labels_) # cluster_ids_x #
        used_paths = []
        for clust in tqdm(range(n_c), desc=f"Selecting images from clusters..."):
            indices = [j for j, x in enumerate(labels) if x == clust]
            selected_index = indices[random.randint(0, len(indices)-1)]
            used_paths.append(new_dataloader.dataset.get_image_path(selected_index))

        return used_paths
