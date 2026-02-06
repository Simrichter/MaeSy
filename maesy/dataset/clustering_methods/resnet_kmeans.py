import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from os import listdir
import os
import random
import json
from torchvision.utils import save_image
from sklearn.cluster import KMeans as skKMeans
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
import argparse
from maesy.dataset import UnlabeledDataset, MultiDataset
from maesy.evaluation import Evaluator
from maesy.evaluation.inferer import Inferer
from maesy.model import ResnetFeatureExtractor, BaseModel

def cluster(paths, n_C=100, batch_size=500, forward_scale=128, step=1, start_index=0):
    # default_path = "C:\\Users\\taran\\CLionProjects\\NDevils2015\\Config\\Logs\\PNGs\\Images\\NDevils2015-Config-1st\\Lower" # The path to the images
    # n_C = 100 # The number of clusters (Decides how many images will be extracted from the dataset)
    # batch_size = 500 # The batch size for passing the data through the neural net.
    # scale = 64 # The size, the images will be downsampled to (Images will become quadratic (scale X scale) )

    # parser = argparse.ArgumentParser(description='abc')
    # parser.add_argument('-p','--path', action="store", dest='path', required=True)
    # parser.add_argument('-c', '--clusters', action="store", dest='clusts', default=n_C)
    # parser.add_argument('-b', '--batchsize', action="store", dest='bs', default=batch_size)
    # parser.add_argument('-s', '--scale', action="store", dest='sc', default=scale)
    # args = parser.parse_args()
    # path = args.path
    # n_C = int(args.clusts)
    # batch_size = args.bs
    # scale = int(args.sc)

    transfs = transforms.Compose([
        transforms.Resize(size=(forward_scale, forward_scale)),
        transforms.ToTensor(),
    ])

    # TODO: Make filetype dynamic (not only supporting .jpg)
    multi_dataset = MultiDataset([UnlabeledDataset(images_dir=path, transforms=transfs, filetype=".jpg", step=step, start_index=start_index) for path in paths])
    multi_dataloader = DataLoader(multi_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model: BaseModel = ResnetFeatureExtractor("resnet50")
    model.eval()

    inferer = Inferer(model=model, data_loader=multi_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    all_predictions, _ = inferer.infer()
    output_cat = torch.cat(all_predictions).to("cpu").detach().numpy()

    print("Clustering")
    batch_size = 1 # batching KMeans is just like running it multiple times on different data
    assert output_cat.shape[0]%batch_size == 0 , f"Error, {output_cat.shape[0]} is not divisible by {batch_size}"
    sk_result = skKMeans(n_clusters=n_C).fit(output_cat.reshape(output_cat.shape[0]//batch_size, -1))
    print("Done")

    labels = list(sk_result.labels_)
    used_paths = []
    for clust in tqdm(range(n_C)):
        indices = [j for j, x in enumerate(labels) if x == clust]
        selected_index = indices[random.randint(0, len(indices)-1)]
        used_paths.append(multi_dataset.get_image_path(selected_index))

    return used_paths