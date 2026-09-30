"""Standalone BeMRS experiment entry point."""
import logging
import os
from pathlib import Path
import shutil
import hydra

ROOT = Path(__file__).resolve().parent
SUPPORTED_SEARCH = ('op_aco', 'mkp_aco', 'cvrp_aco', 'tsp_gls', 'tsp_constructive')

def isolated_workspace(run_dir):
    """Each run owns its candidate modules; datasets remain read-only links."""
    workspace = run_dir / 'workspace'
    for source in (ROOT / 'problems').iterdir():
        if not source.is_dir():
            continue
        target = workspace / 'problems' / source.name
        target.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            destination = target / file.name
            if file.is_file() and file.suffix == '.py':
                shutil.copy2(file, destination)
            elif file.is_dir() and file.name != '__pycache__' and not destination.exists():
                destination.symlink_to(file.resolve(), target_is_directory=True)
    for name in ('prompts', 'utils', 'bemrs'):
        target = workspace / name
        if not target.exists():
            target.symlink_to(ROOT / name, target_is_directory=True)
    return workspace

@hydra.main(version_base=None, config_path='cfg', config_name='config')
def main(cfg):
    from bemrs import BeMRS
    if cfg.problem.problem_name not in SUPPORTED_SEARCH:
        raise ValueError(f'{cfg.problem.problem_name}: evaluator is included, but the server snapshot has no usable BeMRS behavior extractor. Supported search problems: {SUPPORTED_SEARCH}')
    if cfg.api_key:
        os.environ['LITELLM_API_KEY'] = cfg.api_key
    if cfg.base_url:
        os.environ['LITELLM_API_BASE'] = cfg.base_url
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    workspace = isolated_workspace(Path.cwd())
    runner = BeMRS(cfg, workspace)
    if cfg.check:
        from bemrs.core.engine import SearchEngine
        engine = SearchEngine(cfg, runner.problem)
        feature = engine.embedder.encode_single(runner.problem.prompts.seed_func)
        import numpy as np
        assert feature.shape == (engine.embedder.output_dim,)
        assert np.isfinite(feature).all() and not np.all(feature == -1)
        logging.info('CHECK OK: problem=%s dimension=%s archive=%s novelty=%s',
                     cfg.problem.problem_name, len(feature), engine.region_archive_target_distinct,
                     engine.behavior_novelty_slot_enabled)
        return
    if not cfg.api_key:
        raise ValueError('Set OPENAI_API_KEY before starting search')
    code, best_path = runner.evolve()
    Path('best_algorithm.py').write_text(code + '\n')
    logging.info('Search finished. Best algorithm: %s', best_path)

if __name__ == '__main__':
    main()
