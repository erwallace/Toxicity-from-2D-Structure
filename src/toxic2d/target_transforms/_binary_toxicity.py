import numpy as np


class BinaryToxicity:
    def __call__(self, toxicity: np.array) -> int:
        return 1 if np.any(toxicity) else 0
