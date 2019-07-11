# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import numpy as np

try:
    import torch
except ImportError:
    pass


def torch_tensor_from_field(field, init_val=0, cuda=True, requires_grad=False):
    if isinstance(init_val, (int, float)):
        init_val *= np.ones(field.shape, field.dtype.numpy_dtype)
    device = torch.device('cuda' if cuda else 'cpu')
    return torch.tensor(init_val, requires_grad=requires_grad, device=device)
