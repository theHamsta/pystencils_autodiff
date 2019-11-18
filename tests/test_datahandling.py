# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from pystencils_autodiff.framework_integration.datahandling import PyTorchDataHandling


def test_datahandling():
    dh = PyTorchDataHandling((20, 30))

    dh.add_array('foo')


test_datahandling()
