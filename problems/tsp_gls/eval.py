import numpy as np
import logging
import inspect
import random
from gen_inst import TSPInstance, load_dataset, dataset_conf
from gls import TSP_GLS_RANDOM_SEED, guided_local_search
from tqdm import tqdm

# Seed before importing generated code as a guard against import-time
# randomness, then reset again for every evaluated instance in ``solve``.
random.seed(TSP_GLS_RANDOM_SEED)
np.random.seed(TSP_GLS_RANDOM_SEED)

import gpt


def get_heuristics(module):
    """Resolve both BeMRS-generated and hand-written GLS heuristics."""
    possible_names = ["heuristics_v2", "heuristics", "heuristics_v1", "heuristics_v3"]
    for name in possible_names:
        candidate = getattr(module, name, None)
        if inspect.isfunction(candidate):
            if len(inspect.getfullargspec(candidate).args) != 1:
                raise TypeError(
                    f"{name} must accept exactly one argument: distance_matrix"
                )
            return candidate
    raise AttributeError(
        f"No heuristic function found. Expected one of: {possible_names}"
    )


heuristics = get_heuristics(gpt)

perturbation_moves = 30
iter_limit = 1200

def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()

def solve(inst: TSPInstance) -> float:
    # Generated heuristics may use either Python's or NumPy's global RNG.
    # Reset both before candidate execution so repeated evaluations are exact.
    random.seed(TSP_GLS_RANDOM_SEED)
    np.random.seed(TSP_GLS_RANDOM_SEED)

    heu = heuristics(inst.distmat.copy())
    assert tuple(heu.shape) == (inst.n, inst.n)
    result = guided_local_search(inst.distmat, heu, perturbation_moves, iter_limit)
    # print(result)
    return calculate_cost(inst, result)

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    assert mood in ['train', 'val', "test"]

    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
        dataset = load_dataset(dataset_path)

        print(f"[*] Dataset loaded: {dataset_path} with {len(dataset)} instances.")
        
        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)
        
        print("[*] Average:")
        print(np.mean(objs))

    else: # mood == 'val'
        for problem_size in dataset_conf['val']:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
            dataset = load_dataset(dataset_path)
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, instance in enumerate(tqdm(dataset)):
                obj = solve(instance)
                objs.append(obj)
            
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")
