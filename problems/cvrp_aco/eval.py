import os
from aco import ACO, CVRP_ACO_RANDOM_SEED
import sys
import random
import numpy as np
import torch
from scipy.spatial import distance_matrix
import logging
import inspect
import sys
import time
sys.path.insert(0, '../../../')
import gpt

def get_heuristic_name(module, possible_names):
    """Find the generated heuristic without importing LLM-side utilities."""
    for func_name in possible_names:
        candidate = getattr(module, func_name, None)
        if inspect.isfunction(candidate):
            return func_name
    raise AttributeError(f'No heuristic function found. Expected one of: {possible_names}')
possible_func_names = ['heuristics', 'heuristics_v1', 'heuristics_v2', 'heuristics_v3']
heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)
HEURISTIC_ARG_COUNT = len(inspect.getfullargspec(heuristics).args)
N_ITERATIONS = 100
N_ANTS = 30
CAPACITY = 50
DEVICE = os.environ.get('CVRP_ACO_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

def solve(node_pos, demand):
    random.seed(CVRP_ACO_RANDOM_SEED)
    np.random.seed(CVRP_ACO_RANDOM_SEED)
    torch.manual_seed(CVRP_ACO_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CVRP_ACO_RANDOM_SEED)
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1
    if HEURISTIC_ARG_COUNT == 4:
        heu = heuristics(dist_mat.copy(), node_pos.copy(), demand.copy(), CAPACITY) + 1e-09
    elif HEURISTIC_ARG_COUNT == 2:
        heu = heuristics(dist_mat.copy(), demand / CAPACITY) + 1e-09
    else:
        raise TypeError(f'Unsupported heuristic signature with {HEURISTIC_ARG_COUNT} arguments')
    heu[heu < 1e-09] = 1e-09
    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS, device=DEVICE)
    obj = aco.run(N_ITERATIONS)
    return obj
if __name__ == '__main__':
    print('[*] Running ...')
    print(f'[*] ACO device: {DEVICE}')
    start_time = time.perf_counter()
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']
    basepath = os.path.dirname(__file__)
    if not os.path.isfile(os.path.join(basepath, 'dataset/train50_dataset.npy')):
        from gen_inst import generate_datasets
        generate_datasets()
    if mood in ('train', 'val'):
        dataset_path = os.path.join(basepath, f'dataset/{mood}{problem_size}_dataset.npy')
        dataset = np.load(dataset_path)
        demands, node_positions = (dataset[:, :, 0], dataset[:, :, 1:])
        n_instances = node_positions.shape[0]
        print(f'[*] Dataset loaded: {dataset_path} with {n_instances} instances.')
        objs = []
        for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
            obj = solve(node_pos, demand)
            print(f'[*] Instance {i}: {obj}')
            objs.append(obj.item())
        elapsed = time.perf_counter() - start_time
        print(f'[*] Time: {elapsed:.1f}')
        print('[*] Average:')
        print(np.mean(objs))
