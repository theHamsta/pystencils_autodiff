#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import itertools
from typing import Dict

from stringcase import camelcase, pascalcase

import lbmpy_walberla
import pystencils
import pystencils_walberla.codegen
from pystencils.astnodes import Block, EmptyLine
from pystencils.cpu.cpujit import get_headers
from pystencils_autodiff.walberla import (
    AllocateAllFields, CMakeLists, Communication, DefineKernelObjects, DefinitionsHeader, FieldCopy,
    ForLoop, InitBoundaryHandling, LbCommunicationSetup, ResolveUndefinedSymbols, SwapFields,
    SweepCreation, SweepOverAllBlocks, UniformBlockforestFromConfig, WalberlaMain, WalberlaModule)

WALBERLA_MODULES = ["blockforest",
                    "boundary",
                    "communication",
                    "core",
                    "cuda",
                    "domain_decomposition",
                    "executiontree",
                    "fft",
                    "field",
                    "gather",
                    "geometry",
                    "gui",
                    "lbm",
                    "mesa_pd",
                    "mesh",
                    "pde",
                    "pe",
                    "pe_coupling",
                    "postprocessing",
                    "python_coupling",
                    "simd",
                    "sqlite",
                    "stencil",
                    "timeloop",
                    "vtk",
                    "walberla_openvdb"]


class Simulation():
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
                 refinement_scaling=None,
                 boundary_handling_target='gpu',
                 cmake_target_name='autogen_app'):
        self._data_handling = graph_data_handling
        self._lb_rule = lb_rule
        self._refinement_scaling = refinement_scaling
        self._block_forest = UniformBlockforestFromConfig()
        self.parameter_config_block = 'parameters'
        self._codegen_context = codegen_context
        self._boundary_handling = boundary_handling
        self._lb_model_name = 'GeneratedLatticeModel'
        self._flag_field_dtype = 'uint32_t'
        self._kernel_class_generator = Simulation._get_sweep_class_name()
        self._with_gui = False
        self._with_gui_default = False
        self._boundary_kernels = {}
        self._boundary_handling_target = boundary_handling_target
        self._data_handling.merge_swaps_with_kernel_calls()
        self._packinfo_class = 'PackInfo'
        self.cmake_target_name = cmake_target_name

    def _create_helper_files(self) -> Dict[str, str]:

        if self._lb_rule:
            pystencils_walberla.codegen.generate_pack_info_for_field(
                self._codegen_context,
                'PackInfo',
                pystencils.Field.create_generic(self._data_handling.fields['ldc_pdf'].name,
                                                self._data_handling.fields['ldc_pdf'].spatial_dimensions,
                                                self._data_handling.fields['ldc_pdf'].dtype.numpy_dtype,
                                                self._data_handling.fields['ldc_pdf'].index_dimensions,
                                                index_shape=self._data_handling.fields['ldc_pdf'].index_shape,),
                target=self._boundary_handling_target)
            lbmpy_walberla.generate_lattice_model(self._codegen_context, self._lb_model_name,
                                                  self._lb_rule,
                                                  refinement_scaling=self._refinement_scaling)
            if self._boundary_handling:
                for bc in self.boundary_conditions:
                    self._boundary_kernels.update({bc.name: lbmpy_walberla.generate_boundary(
                        self._codegen_context,
                        pascalcase(bc.name),
                        bc,
                        self._lb_rule.method,
                        target=self._boundary_handling_target)},
                    )
                    self._bh_cycler = itertools.cycle(self._boundary_kernels.keys())

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
                'ldc_pdf', field_allocations._cpu_allocations['ldc_pdf']).symbol
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
                                         pdf_field_id,
                                         self._packinfo_class,
                                         self._boundary_handling_target)
                    if self._lb_rule else EmptyLine(),
                    DefineKernelObjects(
                        Block([*call_nodes])
                    )
                ]), self.parameter_config_block
            )
        ])))
        from pystencils_autodiff.framework_integration.printer import DebugFrameworkPrinter

        module.printer = DebugFrameworkPrinter()

        self._codegen_context.write_file("main.cpp", str(module))
        return module

    def _create_defintions_header(self):
        self._codegen_context.write_file("UserDefinitions.h",
                                         str(DefinitionsHeader(self._lb_model_name if self._lb_rule else None,
                                                               self._flag_field_dtype)))

    def _create_cmake_file(self, extra_dependencis=[]):
        walberla_dependencies = []
        import re
        regex = re.compile(r'^["<](\w*)/.*[>$"]')
        headers = get_headers(self._module)
        for h in headers:
            match = regex.match(h)
            if match:
                module = match[1]
                if module in WALBERLA_MODULES:
                    walberla_dependencies.append(module)

        dependencies = walberla_dependencies + extra_dependencis
        try:
            self._codegen_context.write_file("CMakeLists.txt",
                                             str(CMakeLists(self.cmake_target_name,
                                                            [f for f in self._codegen_context.files_written()
                                                             if f.endswith('.cpp') or f.endswith('.cu')],
                                                            depends=dependencies)))
        except AttributeError:
            self._codegen_context.write_file("CMakeLists.txt",
                                             str(CMakeLists(self.cmake_target_name,
                                                            [f for f in self._codegen_context.files.keys()
                                                             if f.endswith('.cpp') or f.endswith('.cu')],
                                                            depends=dependencies)))

    def write_files(self):
        self._create_helper_files()
        self._module = self._create_module()
        self._create_defintions_header()
        self._create_cmake_file()  # has to be called last

    @property
    def boundary_conditions(self):
        return self._boundary_handling._boundary_object_to_boundary_info.keys()

    def _graph_to_sweep(self, c):
        from pystencils_autodiff.graph_datahandling import KernelCall, TimeloopRun, DataTransferKind, DataTransfer

        if isinstance(c, KernelCall):

            if 'indexField' in [f.name for f in c.kernel.ast.fields_accessed]:
                bh = next(self._bh_cycler)
                return f'sweep(blocks, {camelcase(bh)});'

            sweep_class_name = next(self._kernel_class_generator)
            fields_accessed = [f.name for f in c.kernel.ast.fields_accessed]
            c.tmp_field_swaps = list(filter(
                lambda x: x[0].name in fields_accessed and x[1].name in fields_accessed, c.tmp_field_swaps))
            pystencils_walberla.codegen.generate_sweep(
                self._codegen_context, sweep_class_name, c.kernel.ast, field_swaps=c.tmp_field_swaps)
            rtn = SweepOverAllBlocks(SweepCreation(sweep_class_name,
                                                   self._field_allocations,
                                                   c.kernel.ast,
                                                   parameters_to_ignore=[s[1].name for s in c.tmp_field_swaps]),
                                     self._block_forest.blocks)

        elif isinstance(c, TimeloopRun):
            sweeps = [self._graph_to_sweep(s) for s in c.timeloop._single_step_asts]
            rtn = ForLoop(0, c.time_steps, sweeps)

        elif isinstance(c, DataTransfer):
            if c.kind == DataTransferKind.HOST_SWAP:
                src = self._field_allocations._cpu_allocations[c.field.name].symbol
                dst = self._field_allocations._cpu_allocations[c.destination.name].symbol
                rtn = SwapFields(src, dst)
            elif c.kind == DataTransferKind.DEVICE_SWAP:
                src = self._field_allocations._gpu_allocations[c.field.name].symbol
                dst = self._field_allocations._gpu_allocations[c.destination.name].symbol
                rtn = SwapFields(src, dst)
            elif c.kind == DataTransferKind.HOST_TO_DEVICE:
                src = self._field_allocations._cpu_allocations[c.field.name].symbol
                dst = self._field_allocations._gpu_allocations[c.field.name].symbol
                rtn = FieldCopy(self._block_forest.blocks, src, c.field, False, dst, c.field, True)
            elif c.kind == DataTransferKind.DEVICE_TO_HOST:
                src = self._field_allocations._gpu_allocations[c.field.name].symbol
                dst = self._field_allocations._cpu_allocations[c.field.name].symbol
                rtn = FieldCopy(self._block_forest.blocks, src, c.field, True, dst, c.field, False)
            elif c.kind in (DataTransferKind.DEVICE_COMMUNICATION, DataTransferKind.HOST_COMMUNICATION):
                rtn = Communication(self._boundary_handling_target == 'gpu')
            else:
                rtn = None
        else:
            rtn = None

        return rtn
