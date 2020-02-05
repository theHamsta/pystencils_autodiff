#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import os
import sys
from os.path import dirname, expanduser, join

import numpy as np
import sympy as sp

import pystencils
from pystencils.astnodes import Block, EmptyLine, SympyAssignment
from pystencils.data_types import TypedSymbol, create_type
from pystencils_autodiff._file_io import write_file
from pystencils_autodiff.graph_datahandling import GraphDataHandling
from pystencils_autodiff.simulation import Simulation
from pystencils_autodiff.walberla import (
    DefinitionsHeader, FieldAllocation, FlagFieldAllocation, GetParameter, ResolveUndefinedSymbols,
    UniformBlockforestFromConfig, WalberlaMain, WalberlaModule)
from pystencils_walberla.cmake_integration import ManualCodeGenerationContext


def test_walberla():
    x, y = pystencils.fields('x, y:  float32[3d]')
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

        sim = Simulation(dh, ctx)
        print(sim._create_module())


def test_wald_wiesen_lbm():
    import pytest
    pytest.importorskip('lbmpy')
    from lbmpy.creationfunctions import create_lb_collision_rule
    sys.path.append(dirname(__file__))
    with ManualCodeGenerationContext() as ctx:
        from test_graph_datahandling import ldc_setup
        opt_params = {'target': 'gpu'}
        import sympy as sp
        lid_velocity = sp.symbols('lid_velocity')
        lbm_step = ldc_setup(domain_size=(30, 30), optimization=opt_params,
                             fixed_loop_sizes=False, lid_velocity=lid_velocity)

        del lbm_step.data_handling.gpu_arrays.ldc_pdf_tmp

        sim = Simulation(lbm_step.data_handling,
                         ctx,
                         lbm_step.boundary_handling,
                         create_lb_collision_rule(lbm_step.method, optimization=opt_params),
                         cmake_target_name='autogen')
        sim.write_files()

        dir = '/localhome/seitz_local/projects/walberla/apps/autogen/'
        os.makedirs(dir, exist_ok=True)
        for k, v in ctx.files.items():
            with open(join(dir, k), 'w') as file:
                file.write(v)


def test_projection():
    volume = pystencils.fields('volume: float32[3d]')
    projections = pystencils.fields('projection: float32[2d]')
    spacing, foo = sp.symbols('spacing, foo')

    from pystencils_walberla.special_symbols import aabb_min_vector, dx_vector, global_coord
    import pystencils_reco.projection

    from sympy.matrices.dense import matrix_multiply_elementwise
    volume.coordinate_transform = lambda x: aabb_min_vector + matrix_multiply_elementwise(x, dx_vector)
    projections.set_coordinate_origin_to_field_center()
    projections.coordinate_transform *= spacing

    from pystencils.interpolation_astnodes import TextureDeclaration
    TextureDeclaration.headers = []
    projection_matrix = pystencils_reco.matrix_symbols('T', pystencils.data_types.create_type('float32'), 3, 4)
    from pystencils.kernelparameters import FieldPointerSymbol, FieldStrideSymbol

    assignments = pystencils_reco.projection.forward_projection(volume, projections, projection_matrix)
    assignments.main_assignments.append(pystencils.Assignment(foo, FieldPointerSymbol(
        volume.name, volume.dtype, const=True) + FieldStrideSymbol(volume.name, 0) + FieldStrideSymbol(volume.name, 1)))
    assignments.subs({a: b for a, b in zip(pystencils.x_vector(3), global_coord)})
    assignments.kwargs = {}
    print(assignments)
    kernel = assignments.compile('gpu')
    pystencils.show_code(kernel)

    with ManualCodeGenerationContext() as ctx:

        dh = GraphDataHandling((300, 300, 300))
        volume, projections = dh.add_arrays('volume, projection', gpu=True)

        dh.run_kernel(kernel, simulate_only=True)

        sim = Simulation(dh, ctx, cmake_target_name='projection')
        sim._debug = False
        sim.write_files()

        dir = '/localhome/seitz_local/projects/walberla/apps/projection/'
        os.makedirs(dir, exist_ok=True)
        for k, v in ctx.files.items():
            with open(join(dir, k), 'w') as file:
                file.write(v)


def test_global_idx():
    with ManualCodeGenerationContext() as ctx:
        from pystencils_walberla.special_symbols import aabb_min_vector, global_coord

        dh = GraphDataHandling((20, 30, 40))
        my_array = dh.add_array('my_array')

        ast = pystencils.create_kernel([pystencils.Assignment(
            my_array.center, sum(aabb_min_vector))]).compile()
        dh.run_kernel(ast, simulate_only=True)
        dh.save_fields('my_array', expanduser('~/foo'))
        ast = pystencils.create_kernel([pystencils.Assignment(
            my_array.center, sum(global_coord))]).compile()
        dh.run_kernel(ast, simulate_only=True)
        dh.save_fields('my_array', expanduser('~/foo2'))
        # ast = pystencils.create_kernel([pystencils.Assignment(my_array.center, sum(current_global_idx))]).compile()
        # dh.run_kernel(ast, simulate_only=True)

        sim = Simulation(dh, ctx, cmake_target_name='foo')
        sim._debug = False
        sim.write_files()

        dir = '/localhome/seitz_local/projects/walberla/apps/foo/'
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
