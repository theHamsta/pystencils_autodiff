#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import numpy as np
import sympy as sp

import pystencils
from pystencils.astnodes import Block, EmptyLine, SympyAssignment
from pystencils.data_types import TypedSymbol
from pystencils_autodiff._file_io import write_file
from pystencils_autodiff.walberla import (
    DefinitionsHeader, FieldAllocation, GetParameter, PdfFieldAllocation,FlagFieldAllocation,
    UniformBlockForrestFromConfig, WalberlaMain, WalberlaModule)


def test_walberla():
    x, y = pystencils.fields('x, y:  float32[3d]')
    pdf = pystencils.fields('pdf(27):  float32[3d]')
    pdf2 = pystencils.fields('pdf2(27):  float32[3d]')
    flags = pystencils.fields('flag_field:  float32[3d]')

    foo_symbol = TypedSymbol('foo', np.bool)
    number_symbol = TypedSymbol('number', np.float32)
    crazy_plus_one = TypedSymbol('crazy', np.float32)

    block_forrest = UniformBlockForrestFromConfig()

    block = Block([
        block_forrest,
        SympyAssignment(foo_symbol, GetParameter('parameters', foo_symbol)),
        SympyAssignment(number_symbol, GetParameter('parameters', number_symbol, 1.2)),
        SympyAssignment(crazy_plus_one, number_symbol + 1),
        EmptyLine(),
        FieldAllocation(block_forrest.blocks, x),
        PdfFieldAllocation(block_forrest.blocks, pdf, 'LbModel_T'),
        PdfFieldAllocation(block_forrest.blocks, pdf2, 'LbModel_T', [0, 0, 0], 1),
        FlagFieldAllocation(block_forrest.blocks, flags)
    ])

    module = WalberlaModule(WalberlaMain(block))
    code = str(module)
    print(code)

    write_file('/localhome/seitz_local/projects/walberla/apps/autogen/main.cpp', code)

    definitions = DefinitionsHeader(module)
    write_file('/localhome/seitz_local/projects/walberla/apps/autogen/UserDefinitions.h', str(definitions))
