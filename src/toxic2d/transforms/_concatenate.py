from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


class Concatenate:
    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        transformed = [transform(*args, **kwargs) for transform in self.transforms]
        return torch.cat(transformed, dim=0)
