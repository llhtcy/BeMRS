import math
from os import path
import numpy as np
import sys
import argparse
from scipy.spatial import distance_matrix
import logging
from copy import copy
import gpt
select_next_node = next((getattr(gpt, name) for name in
                        ('select_next_node_v2', 'select_next_node_v1', 'select_next_node')
                        if callable(getattr(gpt, name, None))), None)
if select_next_node is None:
    raise ImportError('Candidate must define a select_next_node function')

def eval_heuristic(node_positions: np.ndarray) -> float:
    """
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    
    Returns
    -------
    obj : float
        The length of the generated tour.
    """
    problem_size = node_positions.shape[0]
    dist_mat = distance_matrix(node_positions, node_positions)
    start_node = 0
    solution = [start_node]
    unvisited = set(range(problem_size))
    unvisited.remove(start_node)
    for _ in range(problem_size - 1):
        next_node = select_next_node(current_node=solution[-1], destination_node=start_node, unvisited_nodes=copy(unvisited), distance_matrix=dist_mat.copy())
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f'Node {next_node} is already visited.')
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj
if __name__ == '__main__':
    print('[*] Running ...')
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']
    basepath = path.join(path.dirname(__file__), 'dataset')
    if not path.isfile(path.join(basepath, 'train50_dataset.npy')):
        from gen_inst import generate_datasets
        generate_datasets()
    if mood in ('train', 'val'):
        dataset_path = path.join(basepath, f'{mood}{problem_size}_dataset.npy')
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f'[*] Dataset loaded: {dataset_path} with {n_instances} instances.')
        objs = []
        for i in range(n_instances):
            obj = eval_heuristic(node_positions[i])
            print(f'[*] Instance {i}: {obj}')
            objs.append(obj)
        print('[*] Average:')
        print(np.mean(objs))
