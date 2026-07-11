import torch

import sevenn._keys as KEY
from sevenn.train.loss import PerAtomEnergyLoss


def test_energy_loss_backward_accepts_float64_prediction_and_float32_reference():
    pred = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    batch = {
        KEY.PRED_TOTAL_ENERGY: pred,
        KEY.ENERGY: torch.tensor([1.0], dtype=torch.float32),
        KEY.NUM_ATOMS: torch.tensor([1], dtype=torch.int64),
    }
    loss_def = PerAtomEnergyLoss(criterion=torch.nn.MSELoss())

    loss = loss_def.get_loss(batch)
    loss.backward()

    assert loss.dtype is torch.float64
    assert pred.grad is not None
    assert pred.grad.dtype is torch.float64
