#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import os
import sys
from os.path import dirname, join

import numpy as np

import lbmpy_walberla
import pystencils
from lbmpy.creationfunctions import create_lb_collision_rule, create_lbm_kernel
from pystencils.astnodes import Block, EmptyLine, SympyAssignment
from pystencils.data_types import TypedSymbol, create_type
from pystencils_autodiff._file_io import write_file
from pystencils_autodiff.graph_datahandling import GraphDataHandling
from pystencils_autodiff.walberla import (
    DefinitionsHeader, FieldAllocation, FlagFieldAllocation, GetParameter, PdfFieldAllocation,
    ResolveUndefinedSymbols, UniformBlockforestFromConfig, WalberlaMain, WalberlaModule)
from pystencils_autodiff.wald_und_wiesen_simulation import WaldUndWiesenSimulation
from pystencils_walberla.cmake_integration import ManualCodeGenerationContext


def test_walberla():
    x, y = pystencils.fields('x, y:  float32[3d]')
    pdf = pystencils.fields('pdf(27):  float32[3d]')
    pdf2 = pystencils.fields('pdf2(27):  float32[3d]')
    flags = pystencils.fields('flag_field:  float32[3d]')

    foo_symbol = TypedSymbol('foo', np.bool)
    number_symbol = TypedSymbol('number', np.float32)
    crazy_plus_one = TypedSymbol('crazy', np.float32)

    block_forest = UniformBlockforestFromConfig()

    block = Block([
        block_forest,
        SympyAssignment(foo_symbol, GetParameter('parameters', foo_symbol)),
        SympyAssignment(number_symbol, GetParameter('parameters', number_symbol, 1.2)),
        SympyAssignment(crazy_plus_one, number_symbol + 1),
        EmptyLine(),
        FieldAllocation(block_forest.blocks, x, on_gpu=False),
        PdfFieldAllocation(block_forest.blocks, pdf, 'LbModel_T', on_gpu=True),
        PdfFieldAllocation(block_forest.blocks, pdf2, 'LbModel_T', [0, 0, 0], 1, on_gpu=True),
        FlagFieldAllocation(block_forest.blocks, flags)
    ])

    module = WalberlaModule(WalberlaMain(block))
    code = str(module)
    print(code)

    write_file('/localhome/seitz_local/projects/walberla/apps/autogen/main.cpp', code)

    definitions = DefinitionsHeader(module, 'uint8_t')
    write_file('/localhome/seitz_local/projects/walberla/apps/autogen/UserDefinitions.h', str(definitions))


def test_wald_wiesen_simulation():
    with ManualCodeGenerationContext() as ctx:
        dh = GraphDataHandling((30, 30),
                               periodicity=False,
                               default_ghost_layers=1,
                               default_target='cpu')
        dh.add_arrays('x, y')
        dh.add_arrays('w, z', gpu=True)

        sim = WaldUndWiesenSimulation(dh, ctx)
        print(sim._create_module())


def test_wald_wiesen_lbm():
    sys.path.append(dirname(__file__))
    with ManualCodeGenerationContext() as ctx:
        from test_graph_datahandling import ldc_setup
        opt_params = {'target': 'gpu'}
        lbm_step = ldc_setup(domain_size=(30, 30), optimization=opt_params, fixed_loop_sizes=False)

        sim = WaldUndWiesenSimulation(lbm_step.data_handling,
                                      ctx,
                                      lbm_step.boundary_handling,
                                      create_lb_collision_rule(lbm_step.method, optimization=opt_params))
        sim.write_files()

        dir = '/localhome/seitz_local/projects/walberla/apps/autogen/'
        os.makedirs(dir, exist_ok=True)
        for k, v in ctx.files.items():
            with open(join(dir, k), 'w') as file:
                file.write(v)


def test_resolve_parameters():
    sym = TypedSymbol('s', create_type('double'))
    sym2 = TypedSymbol('t', create_type('double'))

    block_forest = UniformBlockforestFromConfig()

    module = WalberlaModule(WalberlaMain(Block([
        block_forest,
        ResolveUndefinedSymbols(
            Block([
                SympyAssignment(sym, 1 + sym2),
            ]), 'parameters')
    ])))

    print(module)

