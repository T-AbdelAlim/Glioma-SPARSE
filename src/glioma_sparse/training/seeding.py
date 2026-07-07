import os
import random
import numpy as np
import torch


def set_seed(seed, deterministic=True):
    """Seed every RNG that affects training.

    deterministic=True also forces cuDNN into deterministic mode so that two
    runs with the same seed and the same split produce identical results on GPU.
    This is what makes the 5-split variance reflect the data split alone.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optional, stricter: uncomment to make non-deterministic ops raise
        # instead of silently running. Requires the env var below on CUDA.
        # os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Per-worker seeding for DataLoader (standard PyTorch recipe)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    """Seeded generator to hand to DataLoader(generator=...).

    Fixes the shuffle order of the training loader across runs. Without it the
    per-worker seeding is reproducible but the main-process shuffle is not.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
