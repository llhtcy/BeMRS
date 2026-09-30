from aco import ACO, MKP_ACO_RANDOM_SEED
import numpy as np
import torch
import random
import logging
import inspect
import time
import sys
sys.path.insert(0, '../../../')
import gpt

def get_heuristic_name(module, possible_names):
    """Find the generated heuristic without importing LLM-side utilities."""
    for func_name in possible_names:
        if hasattr(module, func_name) and inspect.isfunction(getattr(module, func_name)):
            return func_name
    return None
possible_func_names = ['heuristics', 'heuristics_v1', 'heuristics_v2', 'heuristics_v3']
heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)
N_ITERATIONS = 50
N_ANTS = 10

def solve(prize: np.ndarray, weight: np.ndarray):
    random.seed(MKP_ACO_RANDOM_SEED)
    np.random.seed(MKP_ACO_RANDOM_SEED)
    torch.manual_seed(MKP_ACO_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MKP_ACO_RANDOM_SEED)
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-09
    assert heu.shape == (n,)
    heu[heu < 1e-09] = 1e-09
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)
    obj, _ = aco.run(N_ITERATIONS)
    return obj
if __name__ == '__main__':
    import sys
    import os
    print('[*] Running ...')
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']
    basepath = os.path.dirname(__file__)
    if not os.path.isfile(os.path.join(basepath, f'dataset/train50_dataset.npz')):
        from gen_inst import generate_datasets
        generate_datasets()
    if mood in ('train', 'val'):
        dataset_path = os.path.join(basepath, f'dataset/{mood}{problem_size}_dataset.npz')
        dataset = np.load(dataset_path)
        prizes, weights = (dataset['prizes'], dataset['weights'])
        n_instances = prizes.shape[0]
        print(f'[*] Dataset loaded: {dataset_path} with {n_instances} instances.')
        start_time = time.time()
        objs = []
        for i, (prize, weight) in enumerate(zip(prizes, weights)):
            obj = solve(prize, weight)
            print(f'[*] Instance {i}: {obj}')
            objs.append(obj.item())
        total_time = time.time() - start_time
        print(f'[*] Time: {total_time:.1f}')
        print('[*] Average:')
        print(np.mean(objs))
