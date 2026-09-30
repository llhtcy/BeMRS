"""Public BeMRS runner."""
from .core.search import BeMRS as Search
from .problem_adapter import Problem
from .runtime_config import apply_runtime_config
import os

class BeMRS:
    def __init__(self, cfg, root_dir):
        self.cfg = cfg
        self.root_dir = root_dir
        apply_runtime_config(cfg)
        os.environ['BEMRS_FEATURE_GROUP'] = str(getattr(getattr(cfg.problem, 'behavior', None), 'feature_group', 'full'))
        self.problem = Problem(cfg, str(root_dir))

    def evolve(self):
        return Search(self.cfg, self.problem).run()
