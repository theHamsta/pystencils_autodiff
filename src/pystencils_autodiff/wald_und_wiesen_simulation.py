#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from typing import Dict

import sympy as sp
from stringcase import pascalcase

import lbmpy_walberla
import pystencils
import pystencils_walberla.codegen
from pystencils.astnodes import Block, EmptyLine
from pystencils_autodiff.walberla import (
    AllocateAllFields, CMakeLists, DefinitionsHeader, InitBoundaryHandling, LbCommunicationSetup,
    ResolveUndefinedSymbols, RunTimeLoop, SweepCreation, SweepOverAllBlocks, TimeLoop,
    UniformBlockforestFromConfig, WalberlaMain, WalberlaModule)


class WaldUndWiesenSimulation():
    def _get_sweep_class_name(prefix='Kernel'):

        ctr = 0
        while True:
            yield f'{prefix}{ctr}'
            ctr += 1

    def __init__(self,
                 graph_data_handling,
                 codegen_context,
                 boundary_handling: pystencils.boundaries.BoundaryHandling = None,
                 lb_rule=None,
                 refinement_scaling=None):
        self._data_handling = graph_data_handling
        self._lb_rule = lb_rule
        self._refinement_scaling = refinement_scaling
        self._block_forest = UniformBlockforestFromConfig()
        self.parameter_config_block = 'parameters'
        self._codegen_context = codegen_context
        self._boundary_handling = boundary_handling
        self._lb_model_name = 'GeneratedLatticeModel'
        self._flag_field_dtype = 'uint32_t'
        self._kernel_class_generator = WaldUndWiesenSimulation._get_sweep_class_name()
        self._with_gui = False
        self._with_gui_default = False
        self._boundary_kernels = {}

    def _create_helper_files(self) -> Dict[str, str]:
        if self._lb_rule:
            lbmpy_walberla.generate_lattice_model(self._codegen_context, self._lb_model_name,
                                                  self._lb_rule,
                                                  refinement_scaling=self._refinement_scaling)
            if self._boundary_handling:
                for bc in self.boundary_conditions:
                    self._boundary_kernels.update({bc.name: lbmpy_walberla.generate_boundary(
                        self._codegen_context, pascalcase(bc.name), bc, self._lb_rule.method)})

    def _create_module(self):
        if self._lb_rule:
            lb_shape = (len(self._lb_rule.method.stencil),)
        else:
            lb_shape = (-1,)

        self._field_allocations = field_allocations = AllocateAllFields(self._block_forest.blocks,
                                                                        self._data_handling,
                                                                        lb_shape,
                                                                        self._lb_model_name)

        if self._boundary_handling:
            flag_field_id = field_allocations._cpu_allocations[
                self._boundary_handling.flag_interface.flag_field_name].symbol

        if self._lb_rule:
            pdf_field_id = field_allocations._gpu_allocations.get(
                'ldc_pdfSrc', field_allocations._cpu_allocations['ldc_pdfSrc']).symbol
        else:
            pdf_field_id = None

        call_nodes = filter(lambda x: x, [self._graph_to_sweep(c) for c in self._data_handling.call_queue])

        module = WalberlaModule(WalberlaMain(Block([
            self._block_forest,
            ResolveUndefinedSymbols(
                Block([
                    field_allocations,
                    InitBoundaryHandling(self._block_forest.blocks,
                                         flag_field_id,
                                         pdf_field_id,
                                         self.boundary_conditions,
                                         self._boundary_kernels,
                                         self._field_allocations)
                    if self._boundary_handling else EmptyLine(),
                    LbCommunicationSetup(self._lb_model_name,
                                         pdf_field_id)
                    if self._lb_rule else EmptyLine(),
                    *call_nodes
                ]), self.parameter_config_block)
        ])))

        self._codegen_context.write_file("main.cpp", str(module))
        return module

    def _create_defintions_header(self):
        self._codegen_context.write_file("UserDefinitions.h",
                                         str(DefinitionsHeader(self._lb_model_name, self._flag_field_dtype)))

    def _create_cmake_file(self):
        try:
            self._codegen_context.write_file("CMakeLists.txt",
                                             str(CMakeLists([f for f in self._codegen_context.files_written()
                                                             if f.endswith('.cpp') or f.endswith('.cu')])))
        except AttributeError:
            self._codegen_context.write_file("CMakeLists.txt",
                                             str(CMakeLists([f for f in self._codegen_context.files.keys()
                                                             if f.endswith('.cpp') or f.endswith('.cu')])))

    def write_files(self):
        self._create_helper_files()
        self._create_module()
        self._create_defintions_header()
        self._create_cmake_file()  # has to be called last

    @property
    def boundary_conditions(self):
        return self._boundary_handling._boundary_object_to_boundary_info.keys()

    def _graph_to_sweep(self, c):
        from pystencils_autodiff.graph_datahandling import KernelCall, TimeloopRun

        if isinstance(c, KernelCall):
            sweep_class_name = next(self._kernel_class_generator)
            pystencils_walberla.codegen.generate_sweep(
                self._codegen_context, sweep_class_name, c.kernel.ast)
            rtn = SweepOverAllBlocks(SweepCreation(sweep_class_name, self._field_allocations,
                                                   c.kernel.ast), self._block_forest.blocks)

        elif isinstance(c, TimeloopRun):
            sweeps = []
            for a in c.timeloop._single_step_asts:
                if 'indexField' in [f.name for f in a.fields_accessed]:
                    continue
                sweep_class_name = next(self._kernel_class_generator)
                pystencils_walberla.codegen.generate_sweep(
                    self._codegen_context, sweep_class_name, a)
                sweeps.append(SweepCreation(sweep_class_name, self._field_allocations, a))

            loop = TimeLoop(self._block_forest.blocks, [], sweeps, [], sp.S(c.time_steps))
            rtn = Block([loop, RunTimeLoop(self._block_forest.blocks, loop, self._with_gui, self._with_gui_default)])

        else:
            return None
        return rtn
