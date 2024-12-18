'''Define loss functions that allow mixed outcome types and masked loss computation.'''

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def _mixed_loss(
    pred: torch.tensor, 
    target: torch.tensor, 
    col_types: list, 
    num_adj: float = 1, 
    cat_adj: float = 1, 
    bin_adj: float = 1, 
    pos_adj: float = 1,
):
    '''
    Compute the loss of a batch taking into account the different column types. 
    
    ''' 
    losses = []
    c = 0
    for _, col in enumerate(col_types):
        if col == 'num':
            losses.append(num_adj*MSELoss(reduction = 'none')(pred[:,c], target[:,c]))
            c += 1
        elif col == 'bin':
            losses.append(
                bin_adj*BCEWithLogitsLoss(reduction = 'none')(pred[:,c], target[:,c])
            )
            c += 1
        elif col == 'pos':
            losses.append(
                pos_adj*MSELoss(reduction = 'none')(pred[:,c], target[:,c])
            )
            c += 1
        elif isinstance(col, int):
            losses.append(
                cat_adj*CrossEntropyLoss(reduction = 'none')(pred[:,c:c+col], target[:,c:c+col])
            )
            c += col
        else:
            raise ValueError(f"Unknown column type: {col}")
    return torch.stack(losses, dim = 1)

# write a masked loss function where all outputs are bounded 0-1
def _masked_loss(mixed_losses, mask):
    '''
    
    Consolidate loss for only observed data points.
    
    '''
    if mixed_losses.shape != mask.shape:
        raise ValueError("Mixed losses and mask must have the same shape.")
    loss = torch.sum(mixed_losses*mask.float())/torch.sum(mask)
    return loss
