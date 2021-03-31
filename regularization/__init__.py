# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .regularization import get_regularization, cutmix
from .orthogonal_weight import l2_reg_ortho_32bit
from .loss_fn import get_loss_fn
from .bn_fn import BatchNorm, GhostBatchNorm
