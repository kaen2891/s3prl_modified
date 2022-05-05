# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/hubconf.py ]
#   Synopsis     [ the mockingjay torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
#-------------#
from s3prl.utility.download import _gdriveids_to_filepaths, _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def big_pera_local(ckpt, model_config=None, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (int): -1 (default, the last layer) or an int in range(0, max_layer_num)
    """
    assert os.path.isfile(ckpt)
    if model_config is not None:
        assert os.path.isfile(model_config)
    return _UpstreamExpert(ckpt, model_config, *args, **kwargs)
