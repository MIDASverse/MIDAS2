"""Define loss functions that allow mixed outcome types and masked loss computation."""

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torch.nn.functional as F


class MixedLoss:

    def __init__(
        self, col_types, num_adj=1.0, cat_adj=1.0, bin_adj=1.0, pos_adj=1.0, device=None
    ):
        """
        Initialize the MixedLoss class with column
        """
        self.num_adj = num_adj
        self.cat_adj = cat_adj
        self.bin_adj = bin_adj
        self.pos_adj = pos_adj

        num, bin_, pos, cat = [], [], [], []
        c = 0
        for col in col_types:
            if col == "num":
                num.append(c)
                c += 1
            elif col == "bin":
                bin_.append(c)
                c += 1
            elif col == "pos":
                pos.append(c)
                c += 1
            elif isinstance(col, int):
                cat.append((c, col))
                c += col
            else:
                raise ValueError(f"Unknown column type: {col}")

        self.num_idx = torch.tensor(num, device=device, dtype=torch.long)
        self.bin_idx = torch.tensor(bin_, device=device, dtype=torch.long)
        self.pos_idx = torch.tensor(pos, device=device, dtype=torch.long)
        self.cat_slices = cat  # list of tuples

    @torch.no_grad()
    def _scratch(self, like):
        return like.new_zeros(like.shape)

    def __call__(self, pred, target):

        loss = self._scratch(pred)

        if self.num_idx.numel():
            loss[:, self.num_idx] = self.num_adj * F.mse_loss(
                pred[:, self.num_idx], target[:, self.num_idx], reduction="none"
            )

        if self.bin_idx.numel():
            loss[:, self.bin_idx] = self.bin_adj * F.binary_cross_entropy_with_logits(
                pred[:, self.bin_idx], target[:, self.bin_idx], reduction="none"
            )

        if self.pos_idx.numel():
            loss[:, self.pos_idx] = self.pos_adj * F.mse_loss(
                pred[:, self.pos_idx], target[:, self.pos_idx], reduction="none"
            )

        for start, k in self.cat_slices:
            # target expected as class index, not one-hot:
            tgt = target[:, start : start + k].argmax(1)  # (B,)
            prd = pred[:, start : start + k]  # (B, K)

            loss[:, start] = self.cat_adj * F.cross_entropy(prd, tgt, reduction="none")

        return loss


def _mixed_loss(
    pred: torch.tensor,
    target: torch.tensor,
    col_types: list,
    num_adj: float = 1,
    cat_adj: float = 1,
    bin_adj: float = 1,
    pos_adj: float = 1,
):
    """
    Compute the loss of a batch taking into account the different column types.

    """
    losses = []
    c = 0
    for _, col in enumerate(col_types):
        if col == "num":
            losses.append(num_adj * MSELoss(reduction="none")(pred[:, c], target[:, c]))
            c += 1
        elif col == "bin":
            losses.append(
                bin_adj * BCEWithLogitsLoss(reduction="none")(pred[:, c], target[:, c])
            )
            c += 1
        elif col == "pos":
            losses.append(pos_adj * MSELoss(reduction="none")(pred[:, c], target[:, c]))
            c += 1
        elif isinstance(col, int):
            losses.append(
                cat_adj
                * CrossEntropyLoss(reduction="none")(
                    pred[:, c : c + col], target[:, c : c + col]
                )
            )
            c += col
        else:
            raise ValueError(f"Unknown column type: {col}")
    return torch.stack(losses, dim=1)


# write a masked loss function where all outputs are bounded 0-1
def _masked_loss(mixed_losses, mask):
    """

    Consolidate loss for only observed data points.

    """
    if mixed_losses.shape != mask.shape:
        raise ValueError("Mixed losses and mask must have the same shape.")
    loss = torch.sum(mixed_losses * mask.float()) / torch.sum(mask)
    return loss
