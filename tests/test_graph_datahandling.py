# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import pytest

try:
    from lbmpy.boundaries import UBB, NoSlip
    from lbmpy.lbstep import LatticeBoltzmannStep
    from pystencils_autodiff.graph_datahandling import GraphDataHandling
    from pystencils.slicing import slice_from_direction
    from pystencils_autodiff.computationgraph import ComputationGraph

except ImportError:
    pass

pytest.importorskip('lbmpy')


def create_lid_driven_cavity(domain_size=None, lid_velocity=0.005, lbm_kernel=None,
                             data_handling=None, **kwargs):
    """Creates a lid driven cavity scenario.

    Args:
        domain_size: tuple specifying the number of cells in each dimension
        lid_velocity: x velocity of lid in lattice coordinates.
        lbm_kernel: a LBM function, which would otherwise automatically created
        kwargs: other parameters are passed on to the method, see :mod:`lbmpy.creationfunctions`
        parallel: True for distributed memory parallelization with walberla
        data_handling: see documentation of :func:`create_fully_periodic_flow`
    Returns:
        instance of :class:`Scenario`
    """
    assert domain_size is not None or data_handling is not None
    if data_handling is None:
        optimization = kwargs.get('optimization', None)
        target = optimization.get('target', None) if optimization else None
        data_handling = GraphDataHandling(domain_size,
                                          periodicity=False,
                                          default_ghost_layers=1,
                                          default_target=target)
    step = LatticeBoltzmannStep(data_handling=data_handling,
                                lbm_kernel=lbm_kernel,
                                name="ldc",
                                timeloop_creation_function=data_handling.create_timeloop,
                                **kwargs)

    my_ubb = UBB(velocity=[lid_velocity, 0, 0][:step.method.dim])
    step.boundary_handling.set_boundary(my_ubb, slice_from_direction('N', step.dim))
    for direction in ('W', 'E', 'S') if step.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
        step.boundary_handling.set_boundary(NoSlip(), slice_from_direction(direction, step.dim))

    return step


def ldc_setup(**kwargs):
    ldc = create_lid_driven_cavity(relaxation_rate=1.7, **kwargs)
    ldc.run(50)
    return ldc


def test_graph_datahandling():

    opt_params = {'target': 'gpu', 'gpu_indexing_params': {'block_size': (8, 4, 2)}}
    lbm_step: LatticeBoltzmannStep = ldc_setup(domain_size=(10, 15), optimization=opt_params)
    print(lbm_step._data_handling)

    print(lbm_step._data_handling.call_queue)


def test_graph_generation():

    opt_params = {'target': 'gpu', 'gpu_indexing_params': {'block_size': (8, 4, 2)}}
    lbm_step: LatticeBoltzmannStep = ldc_setup(domain_size=(10, 15), optimization=opt_params)

    graph = ComputationGraph(lbm_step._data_handling.call_queue)
    print("graph.writes: " + str(graph.writes))
    print("graph.reads: " + str(graph.reads))

    print(graph.to_dot())

    graph.to_dot_file('/tmp/foo.dot', with_code=False)
