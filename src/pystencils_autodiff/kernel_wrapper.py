# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

import pystencils.kernel_wrapper
from pystencils_autodiff.framework_integration.printer import FrameworkIntegrationPrinter


class OpWrapper(pystencils.kernel_wrapper.KernelWrapper):
    """
    An extension of :class:`pystencils.kernel_wrapper.KernelWrapper` for operations of
    machine learning frameworks
    """

    def __init__(self, kernel, parameters, ast, forward_ast=None, backward_ast=None):
        self.kernel = kernel
        self.backward_ast = backward_ast
        self.forward_ast = forward_ast
        self.parameters = parameters
        self.ast = ast
        self.forward_ast = forward_ast
        self.num_regs = None

    @property
    def code(self):
        print('das')
        return FrameworkIntegrationPrinter().doprint(self.ast)
