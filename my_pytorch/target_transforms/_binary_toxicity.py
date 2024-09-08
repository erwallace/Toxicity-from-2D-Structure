import numpy as np


class BinaryToxicity:

    def __call__(self, sample: tuple[str, np.array]) -> tuple[str, int]:
        smile, toxicity = sample
        return smile, 1 if np.any(toxicity) else 0
