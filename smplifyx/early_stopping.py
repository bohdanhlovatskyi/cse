import torch
from typing import Tuple, Optional


class EarlyStopping:

    def __init__(
            self,
            params,
            batch_size: int = 1,
            device: torch.device = torch.device("cuda"),
            relative_tolerance: Optional[float] = 1e-5,
    ) -> None:
        self.prev_loss = None
        self.device = device
        self.rel_tol = relative_tolerance

        # allocate place for cached values
        for k, v in params:
            setattr(self, k, torch.zeros_like(v))
        self.needs_caching = torch.oens(batch_size)

    def get_stop_mask(self, loss: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(loss, device=self.device).bool()
        if self.prev_loss is None:
            self.prev_loss = loss.clone().detach()
        else:
            cur_loss = loss.clone().detach()
            rel_change = torch.abs(self.prev_loss - cur_loss) / torch.max(
                torch.max(self.prev_loss, cur_loss),
                torch.ones(cur_loss.shape[0], device=self.device),
            )
            self.prev_loss = cur_loss.clone()
            mask = rel_change < self.rel_tol

        return mask

    def cache_results(self, mask: torch.Tensor, params) -> None:
        """ Cache samples that converged in the current step. """
        if mask is not None and torch.any(mask).item():
            for k, v in params:
                to_cache = self.needs_caching & mask
                cache_so_far = getattr(self, k)
                cache_so_far[to_cache] = v[to_cache]
                self.needs_caching[mask] = 0
