import random

import torch
from sklearn.cluster import KMeans as skKMeans
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from maesy.dataset import UnlabeledDataset, MultiDataset
from maesy.dataset.clustering_methods.pca import pca_reduction
from maesy.evaluation.inferer import Inferer
from maesy.model import ResnetFeatureExtractor, BaseModel


# from kmeans_pytorch import kmeans

def cluster(paths, n_c, batch_size=256, forward_scale=224, step=1, start_index=0):
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

    img_transforms = transforms.Compose([
        transforms.Resize(size=(forward_scale, forward_scale)),
        transforms.ToTensor(),
    ])

    # TODO: Make filetype dynamic (not only supporting .jpg)
    multi_dataset = MultiDataset([UnlabeledDataset(images_dir=path, transforms=img_transforms, step=step, start_index=start_index, use_first_n=10000) for path in paths])
    multi_dataloader = DataLoader(multi_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model: BaseModel = ResnetFeatureExtractor("resnet50")
    model.eval()

    inferer = Inferer(model=model, data_loader=multi_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    all_predictions, _ = inferer.infer()
    output_cat = torch.cat(all_predictions).to("cpu").detach().squeeze()

    out_pca = pca_reduction(output_cat, 256)
    # q = min(256, output_cat.shape[0]-1)
    # if q < 256:
    #     print(f"Warning: Not enough samples for accurate PCA reduction. Reducing to {q} dimensions instead of 256.")
    #
    # U, S, V = torch.pca_lowrank(output_cat, center=True, q=q)
    # out_pca = matmul(output_cat, V)

    print("Clustering")
    start.record()
    # batch_size = 1 # batching KMeans is just like running it multiple times on different data
    # assert out_pca.shape[0]%batch_size == 0 , f"Error, {out_pca.shape[0]} is not divisible by {batch_size}"
    sk_result = skKMeans(n_clusters=n_c).fit(out_pca)
    # cluster_ids_x, cluster_centers = kmeans(X=out_pca.detach(), num_clusters=n_C, distance='euclidean', device=torch.device('cuda:0'), tol=1e-4)
    end.record()
    # torch.cuda.synchronize()
    print(f"Done in {start.elapsed_time(end)/1000:.2f} seconds.")

    labels = list(sk_result.labels_) # cluster_ids_x #
    used_paths = []
    for clust in tqdm(range(n_c), desc=f"Selecting images from clusters..."):
        indices = [j for j, x in enumerate(labels) if x == clust]
        selected_index = indices[random.randint(0, len(indices)-1)]
        used_paths.append(multi_dataset.get_image_path(selected_index))

    return used_paths


# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description='abc')
#     # parser.add_argument('-p','--paths', action="store", dest='paths', required=True, nargs='+')
#     parser.add_argument('-c', '--clusters', action="store", dest='clusts', default=500)
#     parser.add_argument('-b', '--batchsize', action="store", dest='bs', default=256)
#     parser.add_argument('-s', '--scale', action="store", dest='sc', default=224)
#     parser.add_argument('--step', type=int, default=1, help="Step size for sampling images from folders")
#     parser.add_argument('--start-index', type=int, default=0, help="Start index for sampling images from folders")
#     args = parser.parse_args()
#     # paths = args.paths
#     n_C = int(args.clusts)
#     batch_size = args.bs
#     forward_scale = args.sc
#     step = args.step
#     start_index = args.start_index
#
#     used_paths = cluster([r"/media/simon/42A099D63E90C520/Raw Training Data/DutchSalvador/temp/GP3_DutchNaoTeam_Salvador_2025-08-15-14-57-23_out"], n_c=n_C, batch_size=batch_size, forward_scale=forward_scale, step=step, start_index=start_index)
