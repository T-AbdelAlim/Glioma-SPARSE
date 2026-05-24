from collections import Counter
import random


def oversample_paths(paths, labels, shuffle=True, seed=None):
    """
    Oversample dataset so each class has equal number of samples.

    Args:
        paths (list): list of file paths
        labels (list): list of integer labels
        shuffle (bool): whether to shuffle after oversampling
        seed (int or None): for reproducibility

    Returns:
        new_paths, new_labels
    """

    if seed is not None:
        random.seed(seed)

    counts = Counter(labels)
    max_count = max(counts.values())

    new_paths = []
    new_labels = []

    for cls in counts:
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        cls_paths = [paths[i] for i in cls_indices]

        # keep originals
        new_paths.extend(cls_paths)
        new_labels.extend([cls] * len(cls_paths))

        # oversample minority classes
        n_repeat = max_count - len(cls_paths)
        if n_repeat > 0:
            extra = random.choices(cls_paths, k=n_repeat)
            new_paths.extend(extra)
            new_labels.extend([cls] * n_repeat)

    # optional shuffle (recommended)
    if shuffle:
        combined = list(zip(new_paths, new_labels))
        random.shuffle(combined)
        new_paths, new_labels = zip(*combined)
        new_paths = list(new_paths)
        new_labels = list(new_labels)

    return new_paths, new_labels


def get_class_distribution(labels, class_names=None):
    """
    Get class counts (for debugging/logging).

    Args:
        labels (list): list of integer labels
        class_names (list or None): optional list of class names

    Returns:
        dict: {class_name or index: count}
    """

    counts = Counter(labels)

    if class_names is not None:
        return {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}

    return dict(counts)