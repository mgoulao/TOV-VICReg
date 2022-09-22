from collections import deque
import random
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE, Isomap
import matplotlib.pyplot as plt
from matplotlib import cm, colors as matplot_colors
from dataset.dqn_dataset import MultiDQNReplayDataset, _get_DQN_Replay_loader

def is_flicker_frame(obs):
    threshold = 0.99
    torch.count_nonzero(obs) / torch.numel(obs) > threshold

def get_datapoints(data_loaders, n_datapoints_per_game):
    data_points = []
    for data_loader in data_loaders:
        random_indexes = []
        n_index = 0
        while n_index < n_datapoints_per_game:
            rnd_num = int(random.random()*len(data_loader))
            if not rnd_num in random_indexes:
                random_indexes.append(rnd_num)
                n_index += 1
        i = 0
        n = 0
        for obj in data_loader:
            if i in random_indexes:
                n += 1
                if not is_flicker_frame(obj[0]):
                    data_points.append(obj[0])
            if n >= n_datapoints_per_game or i >= len(data_loader):
                break
            i += 1
    return torch.cat(data_points).cuda()

def get_sequential_datapoints(datasets, start_indx, n_datapoints_per_game, skip=0):
    data_points = []
    for dataset in datasets:
        for i in range(0, n_datapoints_per_game * (skip + 1), skip+1):
            if i % 3 == 0 and not is_flicker_frame(dataset[i + start_indx][0]):
                data_points.append(dataset[i + start_indx][0])
    return torch.stack(data_points).cuda()

def get_sequence_similarities(datapoints, sequences_size, n_games, n_datapoints_per_game):
    seq = []
    game_seq = []
    game = -1
    prev = None
    for i in range(sequences_size * n_games):
        if i % sequences_size == 0:
            if i > 1:
                seq.append(game_seq)
            game_seq = []
            game += 1
            prev = None
        curr = datapoints[i % n_datapoints_per_game + n_datapoints_per_game * game]

        if prev is not None:
            game_seq.append(calculate_cosine_similarity(prev, curr))
        prev = curr


    return seq


def get_data_loaders(games):
    data_loaders = []
    obs_transform = transforms.Compose([
                    transforms.ConvertImageDtype(torch.float)
                ])
    for game in games:
        data_loaders.append(_get_DQN_Replay_loader(Path("/media/msgstorage/dqn"), [game], ["1"], 3, 10000, 1, 1, obs_transform))

    return data_loaders

def get_datasets(games, size=10000, actions=False, checkpoint="1"):
    datasets = []
    obs_transform = transforms.Compose([
                    transforms.ConvertImageDtype(torch.float)
                ])
    for game in games:
        datasets.append(MultiDQNReplayDataset(Path("/media/msgstorage/dqn"), [game], [checkpoint], 3, size, obs_transform, actions=actions))

    return datasets

def calculate_cosine_similarity(a, b):
    assert a.shape == b.shape
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a*b).sum(dim=-1).item()

def create_similarity_matrix(representations, min=False):
    matrix = []
    n_datapoints = len(representations)
    for i in range(n_datapoints):
        row = []
        for j in range(n_datapoints):
            if j < i and min:
                similarity = -1.0
            else:
                similarity = calculate_cosine_similarity(representations[i], representations[j])
            row.append(similarity)
        matrix.append(row)

    return matrix

def calculate_tsne(representations):
    tsne = TSNE(n_components=2, verbose=1, n_iter=300, learning_rate='auto', init='pca')
    return tsne.fit_transform(representations)

def calculate_umap(representations):
    return umap.UMAP().fit_transform(representations)

def calculate_isomap(representations):
    isomap = Isomap(n_components=2, n_neighbors=10)
    return isomap.fit_transform(representations)

def plot_tsne(tsne_results, sources_names=[""]):
    n_sources = len(sources_names)
    step = tsne_results.shape[0] // n_sources
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(n_sources):
        ax.scatter(tsne_results[step*i:step*(i+1)+1, 0], tsne_results[step*i:step*(i+1)+1, 1], label=f"{sources_names[i]}", c=i)
    ax.legend()
    plt.show()

def get_2d_plot(tsne_results, sources_names=[""]):
    n_sources = len(sources_names)
    step = tsne_results.shape[0] // n_sources
    color_map = cm.get_cmap("GnBu")
    norm = matplot_colors.Normalize(vmin=-1, vmax=n_sources-1)
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(n_sources):
        ax.scatter(tsne_results[step*i:step*(i+1), 0], tsne_results[step*i:step*(i+1), 1],
                    label=sources_names[i],
                    c=color_map(norm([i for _ in range(step)])))
    ax.legend()
    return fig


def get_2d_seq_points_plot(points, n_steps):
    step = points.shape[0] // n_steps
    color_map = cm.get_cmap("nipy_spectral")
    alphas = list(np.arange(1, n_steps + 1) / n_steps)
    norm = matplot_colors.Normalize(vmin=-1, vmax=step)
    time_colors = color_map(norm(np.arange(step)))
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(n_steps):
        ax.scatter(points[step*i:step*(i+1), 0], points[step*i:step*(i+1), 1],
                    c=time_colors, alpha=alphas[i])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax)
    return fig


def plot_visualization_experiments(representations_list, sources_names=[], tsne=False, isomap=False):
    representations = np.concatenate(representations_list)
    if tsne:
        print("t-SNE:")
        tsne_results = calculate_tsne(representations)
        plot_tsne(tsne_results, sources_names)
    if isomap:
        print("Isomap:")
        isomap_results = calculate_isomap(representations)
        plot_tsne(isomap_results, sources_names)


def plot_evolution(points, step_map, sources_names):
    fig, ax = plt.subplots(figsize=(15, 7))
    color_maps = [cm.get_cmap("GnBu"), cm.get_cmap("Oranges"),  cm.get_cmap("Greys"), cm.get_cmap("YlGn"), cm.get_cmap("PuRd")]
    step = 0
    for i in range(len(step_map)):
        curr_step_map = step_map[i]
        norm = matplot_colors.Normalize(vmin=-1, vmax=curr_step_map[0]-1)
        color_map = color_maps[i]
        for j in range(curr_step_map[0]):
            size = curr_step_map[1]
            label = None
            if j == curr_step_map[0] - 1:
                label = sources_names[i]
            ax.scatter(points[step:step+size, 0], points[step:step+size, 1], label=label, c=color_map(norm([j for _ in range(size)])))
            step += size
    ax.legend()
    plt.show()


def plot_evolution_visualization_experiments(representations_list, step_map, sources_names=[], tsne=False, isomap=False):
    representations = np.concatenate(representations_list)
    if tsne:
        print("t-SNE:")
        tsne_results = calculate_tsne(representations)
        plot_evolution(tsne_results, step_map, sources_names)
    if isomap:
        print("Isomap:")
        isomap_results = calculate_isomap(representations)
        plot_evolution(isomap_results, step_map, sources_names)


def plot_experiments(representations, n_datapoints_per_game=10, similarity=False):
    if similarity:
        print("Similarity:")
        similarity_matrix = create_similarity_matrix(representations)
        min_similarity_matrix = create_similarity_matrix(representations, min=True)
        game_similarity_matrix = create_similarity_matrix(representations[2*n_datapoints_per_game:3*n_datapoints_per_game])
        fig, axs = plt.subplots(1,3, figsize=(13, 5))
        mat1 = axs[0].matshow(similarity_matrix)
        mat2 = axs[1].matshow(min_similarity_matrix)
        mat3 = axs[2].matshow(game_similarity_matrix)
        fig.colorbar(mat1, ax=axs[0])
        fig.colorbar(mat2, ax=axs[1])
        fig.colorbar(mat3, ax=axs[2])
        plt.show()
