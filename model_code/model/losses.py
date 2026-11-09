# Training loss: one term per output head.
#   mode       cross-entropy on every row
#   direction  cross-entropy only on rows with a direction label (target / trail); -1 is ignored
#   pick, drop binary cross-entropy

import torch.nn.functional as F


def compute_loss(out, batch, tcfg):
    l_mode = F.cross_entropy(out["mode_logits"], batch["mode"])
    l_dir = F.cross_entropy(out["dir_logits"], batch["direction"], ignore_index=-1)
    l_pick = F.binary_cross_entropy_with_logits(out["pick_logit"], batch["pick"])
    l_drop = F.binary_cross_entropy_with_logits(out["drop_logit"], batch["drop"])
    total = (tcfg.w_mode * l_mode + tcfg.w_dir * l_dir
             + tcfg.w_pick * l_pick + tcfg.w_drop * l_drop)
    return total, dict(loss=total, loss_mode=l_mode, loss_dir=l_dir, loss_pick=l_pick, loss_drop=l_drop)
