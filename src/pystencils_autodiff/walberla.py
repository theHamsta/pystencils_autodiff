#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


from abc import ABC
from os.path import dirname, join

import jinja2
import sympy as sp
from stringcase import camelcase, pascalcase

import pystencils
from pystencils.astnodes import SympyAssignment
from pystencils.data_types import TypedSymbol, create_type
from pystencils_autodiff._file_io import read_template_from_file
from pystencils_autodiff.framework_integration.astnodes import JinjaCppFile


def _make_field_type(field, on_gpu):
    from pystencils_walberla.jinja_filters import make_field_type, get_field_fsize
    f_size = get_field_fsize(field)

    return make_field_type(pystencils.data_types.get_base_type(field.dtype), f_size, on_gpu)


class FieldType(JinjaCppFile):

    TEMPLATE = jinja2.Template("{{ field_type }}")

    def __init__(self, field: pystencils.Field, on_gpu: bool):

        field_type = _make_field_type(field, on_gpu)

        ast_dict = {'on_gpu': on_gpu,
                    'field_type': field_type
                    }
        JinjaCppFile.__init__(self, ast_dict)

    @property
    def headers(self):
        if self.ast_dict.on_gpu:
            return ['"field/GhostLayerField.h"']
        else:
            return ['"cuda/GPUField.h"']


class WalberlaModule(JinjaCppFile):
    TEMPLATE = read_template_from_file(join(dirname(__file__), 'walberla_main.tmpl.cpp'))

    def __init__(self, main):
        ast_dict = {'main': main}
        JinjaCppFile.__init__(self, ast_dict)


class CMakeLists(JinjaCppFile):
    TEMPLATE = read_template_from_file(join(dirname(__file__), 'CMakeLists.tmpl.txt'))

    def __init__(self, files_to_compile):
        ast_dict = {'files_to_compile': files_to_compile}  # we try to avoid evil globbing
        JinjaCppFile.__init__(self, ast_dict)


class WalberlaMain(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
int main( int argc, char ** argv )
{
   using namespace walberla;
   using namespace walberla_user;
   using namespace walberla::pystencils;

   Environment {{ walberl_env }}( argc, argv );

   {{ body | indent(3) }}

   return EXIT_SUCCESS;
}
    """)

    headers = ['"core/Environment.h"',
               '<cstdlib>',
               '"UserDefinitions.h"']

    def __init__(self, body, walberl_env="walberlaEnv"):

        ast_dict = {
            'body': body,
            'walberl_env': TypedSymbol(walberl_env, "Environment")
        }

        super().__init__(ast_dict)

    @property
    def symbols_defined(self):
        return {self.ast_dict.walberl_env}


class BlockForestCreation(JinjaCppFile):
    TEMPLATE = jinja2.Template("""auto {{ blocks }} = walberla_user::createBlockForest(walberlaEnv);""")
    def __init__(self, block_forest_name):

        ast_dict = {
            'blocks': TypedSymbol(block_forest_name, "auto")
        }

        super().__init__(ast_dict)

    @property
    def symbols_defined(self):
        return {self.ast_dict.blocks}


class UniformBlockforestFromConfig(BlockForestCreation):
    TEMPLATE = jinja2.Template(
        """auto {{ blocks }} = blockforest::createUniformBlockGridFromConfig( walberlaEnv.config() );""")
    headers = ['"blockforest/Initialization.h"']

    def __init__(self):
        super().__init__('blocks')

    @property
    def config_required(self):
        return {"DomainSetup": {"blocks": [1, 1, 1], "cellsPerBlock": [100, 100, 100]}}

    @property
    def blocks(self):
        return self.ast_dict.blocks


class DefinitionsHeader(JinjaCppFile):
    TEMPLATE = read_template_from_file(join(dirname(__file__), 'walberla_user_defintions.tmpl.hpp'))

    def __init__(self, lb_model_name, flag_field_type):
        self.headers = ['<cstdint>', '"lbm/field/PdfField.h"', f'"{lb_model_name}.h"', '"field/FlagField.h"']
        super().__init__({'lb_model_name': lb_model_name, 'flag_field_type': flag_field_type})


class Using(JinjaCppFile):
    TEMPLATE = jinja2.Template("using {{ new_type }} = {{ definition }};\n")

    def __init__(self, new_type, definition):
        ast_dict = {
            'new_type': new_type,
            'definition': definition,
        }
        super().__init__(ast_dict)

    @property
    def symbols_defined(self):
        return {self.ast_dict.new_type}


class GetParameter(JinjaCppFile):

    TEMPLATE = jinja2.Template(
        'walberlaEnv.config()->getOneBlock("{{ block }}").getParameter<{{ key.dtype }}>("{{ key }}"{% if default %}, static_cast<{{ key.dtype }}>({{ default }}){% endif %})'  # noqa
        )

    def __init__(self, block: str, key, default_value=None):
        ast_dict = {
            'key': key,
            'block': block,
            'default': default_value
        }
        super().__init__(ast_dict)

    @property
    def config_required(self):
        return {self.ast_dict.block: {self.ast_dict.symbol: self.ast_dict.default_value}}

    def __sympy__(self):
        return TypedSymbol(self.ast_dict.key.name + "Config", str(self.ast_dict.key.dtype))

    def __getattr__(self, name):
        return getattr(self.__sympy__(), name)


class FieldAllocation(JinjaCppFile):
    """
    .. code:: cpp

        BlockDataID addToStorage(const shared_ptr < BlockStorage_T > & blocks,
                                  const std:: string & identifier,
                                  const typename GhostLayerField_T:: value_type & initValue = typename GhostLayerField_T: : value_type(),
                                  const Layout layout=zyxf,
                                  const uint_t nrOfGhostLayers=uint_t(1),
                                  const bool alwaysInitialize=false,
                                  const std:: function < void (GhostLayerField_T * field, IBlock * const block ) > & initFunction =
                                     std: : function < void (GhostLayerField_T * field, IBlock * const block ) > (),
                                  const Set < SUID > & requiredSelectors = Set < SUID > : : emptySet(),
                                  const Set < SUID > & incompatibleSelectors = Set < SUID > : : emptySet() )

    """  # noqa

    TEMPLATE = jinja2.Template("""
{%- if on_gpu -%}
BlockDataID {{ field_name }}_data_gpu = cuda::addGPUFieldToStorage<{{ field_type }}>({{ block_forest }},
                                    "{{ field_name }}",
                                    {{ f_size }},
                                    {{ layout }},
                                    {{ num_ghost_layers }},
                                    {{ usePitchedMem }} );
{%- else -%}
BlockDataID {{ field_name }}_data = field::addToStorage<{{ field_type }}>( {{ block_forest }},
                              "{{ field_name }}"
     {%- if init_value -%}      , {{ init_value }}{% endif %}
     {%- if layout -%}      , {{ layout }}{% endif %}
     {%- if num_ghost_layers -%}, {{ num_ghost_layers }} {% endif %}
     {%- if always_initialize -%}, {{ always_initialize }} {% endif %});
{%- endif %}
""")  # noqa

    def __init__(self, block_forest, field, on_gpu=False, usePitchedMem=True, num_ghost_layers=1):
        self._symbol = TypedSymbol(field.name + ('_data_gpu' if on_gpu else '_data'), 'BlockDataID')
        ast_dict = {
            'block_forest': block_forest,
            'field_name': field.name,
            # f'GhostLayerField< {field.dtype}, {field.index_shape[0] if field.index_shape else 1} >'
            'field_type': FieldType(field, on_gpu),
            'on_gpu': on_gpu,
            'f_size': field.index_shape[0] if field.index_shape else 1,
            'init_value': f'{field.dtype}{{}}',
            'num_ghost_layers': num_ghost_layers,
            'layout': 'field::zyxf',
            'usePitchedMem': 'true' if usePitchedMem else 'false',
        }
        super().__init__(ast_dict)

    @property
    def symbol(self):
        return self._symbol

    @property
    def symbols_defined(self):
        return {self.symbol}

    @property
    def headers(self):
        return (['"cuda/AddGPUFieldToStorage.h"', '"field/GhostLayerField.h"']
                if self.ast_dict.on_gpu
                else ['"field/AddToStorage.h"'])


class WalberlaVector(JinjaCppFile):
    TEMPLATE = jinja2.Template("""math::Vector{{ndim}}<{{dtype}}>({{offsets}})""")

    def __init__(self, offset, dtype='real_t'):
        ast_dict = {
            'offset': offset,
            'offsets': ', '.join(str(o) for o in offset),
            'dtype': dtype,
            'ndim': len(offset),
        }
        super().__init__(ast_dict)

    def __sympy__(self):
        return sp.Matrix(self.ast_dict.offset)


class PdfFieldAllocation(FieldAllocation):
    """
    .. code: : cpp

        BlockDataID addPdfFieldToStorage(const shared_ptr < BlockStorage_T > & blocks, const std:: string & identifier,
                                          const LatticeModel_T & latticeModel,
                                          const Vector3 < real_t > & initialVelocity, const real_t initialDensity,
                                          const uint_t ghostLayers,
                                          const field:: Layout & layout = field: : zyxf,
                                          const Set < SUID > & requiredSelectors     = Set < SUID > : : emptySet(),
                                          const Set < SUID > & incompatibleSelectors = Set < SUID > : : emptySet() )
    """
    TEMPLATE = jinja2.Template("""BlockDataID {{field_name}}_data = lbm::field::addPdfFieldToStorage < {{ field_type }} > ({{ block_forest }},
                              {{field_name}},
                              {{lattice_model}}
     {%- if initial_velocity -%}      , {{initial_velocity }} {% endif %}
     {%- if initial_density -%}      , {{initial_density }} {% endif %}
     {%- if num_ghost_layers -%}, {{num_ghost_layers }} {% endif %});
""")  # noqa

    def __init__(self, block_forest, field, lattice_model, initial_velocity=None, initial_density=None, on_gpu=False):
        super().__init__(block_forest, field, on_gpu)
        if initial_velocity and not isinstance(initial_velocity, WalberlaVector):
            initial_velocity = WalberlaVector(initial_velocity)

        self.ast_dict.update({
            'initial_density': initial_density,
            'lattice_model': lattice_model,
            'initial_velocity': initial_velocity,
        })

    headers = ['"lbm/field/AddToStorage.h"']


class FlagFieldAllocation(FieldAllocation):
    """
    .. code: : cpp

        template< typename FlagField_T, typename BlockStorage_T >
        BlockDataID addFlagFieldToStorage( const shared_ptr< BlockStorage_T > & blocks,
                                           const std::string & identifier,
                                           const uint_t nrOfGhostLayers = uint_t(1),
                                           const bool alwaysInitialize = false,
                                           const std::function< void ( FlagField_T * field, IBlock * const block ) > & initFunction =
                                              std::function< void ( FlagField_T * field, IBlock * const block ) >(),
                                           const Set<SUID> & requiredSelectors = Set<SUID>::emptySet(),
                                           const Set<SUID> & incompatibleSelectors = Set<SUID>::emptySet() )
    """  # noqa
    TEMPLATE = jinja2.Template("""BlockDataID {{field_name}}_data = field::addFlagFieldToStorage < {{ field_type }} > ({{ block_forest }},
                              {{field_name}}
     {%- if num_ghost_layers -%}, {{num_ghost_layers }} {% endif %});
""")  # noqa

    def __init__(self, block_forest, field, on_gpu=False):
        super().__init__(block_forest, field, on_gpu)

    headers = ['"field/AddToStorage.h"']


class FlagUidDefinition(JinjaCppFile):
    TEMPLATE = jinja2.Template('const FlagUID {{ name }}FlagUID("{{ name }}");')

    def __init__(self, name):
        self._symbol = TypedSymbol(name + 'FlagUID', 'FlagUID')
        ast_dict = {
            'name': name,
        }
        super().__init__(ast_dict)

    headers = ['"field/FlagUID.h"']

    @property
    def symbol(self):
        return self._symbol

    @property
    def symbols_defined(self):
        return {self.symbol}


class BoundaryHandling(ABC):
    pass


class BoundaryHandlingFromConfig(JinjaCppFile):

    TEMPLATE = jinja2.Template("""auto {{  boundaries_config }} = walberlaEnv.config()->getOneBlock( "Boundaries" );
geometry::initBoundaryHandling<FlagField_T>(*{{ block_forest }}, {{ flag_field_id }}, {{ boundaries_config }});
geometry::setNonBoundaryCellsToDomain<FlagField_T>(*{{ block_forest }}, {{ flag_field_id }}, {{ fluid_uid }});""")

    def __init__(self, block_forest, flag_field_id, fluid_uid):
        self._symbol = boundaries_config = TypedSymbol('boundariesConfig', 'auto')
        ast_dict = {
            'flag_field_id': flag_field_id,
            'block_forest': block_forest,
            'fluid_uid': fluid_uid,
            'boundaries_config': boundaries_config,
        }
        super().__init__(ast_dict)

    @property
    def symbol(self):
        return self._symbol

    @property
    def symbols_defined(self):
        return {self.symbol}


class FillFromFlagField(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
    {{ boundary_condition }}.fillFromFlagField<{{ flag_field_type }}>( {{ block_forest }}, {{ flag_field_id }}, FlagUID("{{ boundary_name }}"), {{ flag_field_id }});
    """)  # noqa

    def __init__(self, flag_field_id, fluid_uid):
        self._symbol = boundaries_config = TypedSymbol('boundariesConfig', 'auto')
        ast_dict = {
            'flag_field_id': flag_field_id,
            'fluid_uid': fluid_uid,
            'boundaries_config': boundaries_config,
        }
        super().__init__(ast_dict)

    @property
    def symbol(self):
        return self._symbol

    @property
    def symbols_defined(self):
        return {self.symbol}


class LbCommunicationSetup(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
{{CommonicationScheme}}<lbm::{{ lb_model_type }}::CommunicationStencil> {{ communication }}( blocks );
{{ communication }}.addPackInfo( make_shared< {{ packinfo_class }} >( {{ pdf_id }} ) );
 """)  # noqa

    def __init__(self, lb_model_type, pdf_id, packinfo_class, gpu):
        self._symbol = TypedSymbol('communication', 'auto')
        self.gpu = gpu

        ast_dict = {
            'lb_model_type': lb_model_type,
            'pdf_id': pdf_id,
            'communication': self._symbol,
            'packinfo_class': packinfo_class,
            'CommonicationScheme': ('cuda::communication::UniformGPUScheme'
                                    if gpu else 'blockforest::communication::UniformBufferedScheme')
        }
        super().__init__(ast_dict)

    @property
    def symbol(self):
        return self._symbol

    @property
    def symbols_defined(self):
        return {self.symbol}

    @property
    def headers(self):
        if self.gpu:
            return ['"cuda/communication/UniformGPUScheme.h"', '"lbm/communication/PdfFieldPackInfo.h"', '"PackInfo.h"']
        else:
            return ['"blockforest/communication/UniformBufferedScheme.h"',
                    '"lbm/communication/PdfFieldPackInfo.h"',
                    '"PackInfo.h"']


class BeforeFunction(JinjaCppFile):

    TEMPLATE = jinja2.Template("""BeforeFunction( {{ f }}, "{{ f }}" )""")  # noqa

    def __init__(self, function):
        ast_dict = {
            'f': function,
        }
        super().__init__(ast_dict)


class AfterFunction(JinjaCppFile):

    TEMPLATE = jinja2.Template("""AfterFunction( {{ f }}, "{{ f }}" )""")  # noqa

    def __init__(self, function):
        ast_dict = {
            'f': function,
        }
        super().__init__(ast_dict)


class Sweep(JinjaCppFile):

    TEMPLATE = jinja2.Template("""Sweep( {{ f }}, "{{ f }}" )""")  # noqa

    def __init__(self, function):
        ast_dict = {
            'f': function,
        }
        super().__init__(ast_dict)


class TimeLoop(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
{{ timeloop_definition }}

{% for f in before_functions -%}
{{timeloop_name}}.add() << {{ f }};
{% endfor %}
{% for s in sweeps -%}
{{timeloop_name}}.add() << {{ s }};
{% endfor %}
{% for f in after_functions -%}
{{timeloop_name}}.add() << {{ f }};
{% endfor %}
    """)  # noqa

    def __init__(self, block_forest, before_functions, sweeps, after_functions, timesteps, timeloop_name='timeloop'):
        self._symbol = TypedSymbol(timeloop_name, 'SweepTimeloop')
        ast_dict = {
            'timeloop_definition': CppObjectConstruction(self._symbol,
                                                         [f'{block_forest}->getBlockStorage()',
                                                          timesteps]),
            'before_functions': [BeforeFunction(f) for f in before_functions],
            'sweeps': [Sweep(f) for f in sweeps],
            'after_functions': [AfterFunction(f) for f in after_functions],
            'timeloop_name': timeloop_name
        }
        super().__init__(ast_dict)

    @property
    def symbol(self):
        return self._symbol

    @property
    def symbols_defined(self):
        return {self.symbol}

    headers = ['"timeloop/all.h"']


class ForLoop(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
for( {{loop_symbol.dtype}} {{loop_symbol}} = {{loop_start}}; {{loop_symbol}} < {{loop_end}}; {{loop_symbol}} += {{loop_increment}} ) {
   {%- for c in children %}
   {{ c | indent(3) -}}
   {% endfor %}
}
""")  # noqa

    def __init__(self,
                 loop_start,
                 loop_end,
                 children,
                 loop_symbol=TypedSymbol('t_', create_type('int64')),
                 loop_increment=1):
        ast_dict = {
            'loop_symbol': loop_symbol,
            'loop_start': loop_start,
            'loop_end': loop_end,
            'children': children,
            'loop_increment': loop_increment
        }
        super().__init__(ast_dict)

    @property
    def loop_symbol(self):
        return self.ast_dict.loop_symbol

    @property
    def symbols_defined(self):
        return {self.loop_symbol}


class U_Rho_Adaptor(JinjaCppFile):
    """Docstring for U_Rho_Adaptor. """

    TEMPLATE = jinja2.Template("""
field::addFieldAdaptor<lbm::Adaptor<lbm::{{ lb_model_type }}>::Density>       ( {{ block_forest }}, {{ pdf_id }}, "{{ density_name }}" );
field::addFieldAdaptor<lbm::Adaptor<lbm::{{ lb_model_type }}>::VelocityVector>( {{ block_forest }}, {{ pdf_id }}, "{{ velocity_name }}" );
    """)  # noqa

    def __init__(self, block_forest, lb_model_type, pdf_id):
        self.density_name = "DensityAdaptor"
        self.velocity_name = "VelocityAdaptor"
        ast_dict = {
            'lb_model_type': lb_model_type,
            'pdf_id': pdf_id,
            'block_forest': block_forest,
            'density_name': self.density_name,
            'velocity_name': self.velocity_name,
        }
        super().__init__(ast_dict)

    headers = ['"timeloop/all.h"']


class RunTimeLoop(JinjaCppFile):
    TEMPLATE = jinja2.Template("""{%- if with_gui == 'true' %}
if( parameters.getParameter<bool>( "useGui", {{ use_gui_default }}) )
{
   GUI gui ( {{ timeloop }}, {{ block_forest }}, argc, argv );
   lbm::connectToGui<LatticeModel_T> ( gui );
   gui.run();
}
else {
   {{ timeloop }}.run();
}
{%- else %}
timeloop.run();
{%- endif %}""")  # noqa

    def __init__(self, block_forest, timeloop, with_gui=False, use_gui_default='false'):
        self.density_name = "DensityAdaptor"
        self.velocity_name = "VelocityAdaptor"
        ast_dict = {
            'block_forest': block_forest,
            'timeloop': timeloop.symbol,
            'with_gui': with_gui,
            'use_gui_default': use_gui_default,
        }
        super().__init__(ast_dict)

    @property
    def headers(self):
        if self.ast_dict.with_gui:
            return [
                "gui/all.h",
                "lbm/gui/Connection.h"
            ]
        else:
            return []


class CppObjectConstruction(JinjaCppFile, pystencils.astnodes.SympyAssignment):

    TEMPLATE = jinja2.Template("""{{  symbol.dtype  }} {{ symbol }}({{ arg_list }});""")  # noqa

    def __init__(self, symbol, args):
        JinjaCppFile.__init__(self, {})
        self.ast_dict.update({
            'symbol': symbol,
            'args': args,
            'arg_list': ', '.join(self.printer(a) for a in args)
        })
        pystencils.astnodes.SympyAssignment.__init__(self, symbol, 1)


class ResolveUndefinedSymbols(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
{% for p in parameters %}
{{ p }}
{% endfor %}
{{ block  }}
""")  # noqa

    def __init__(self, block, config_block):
        self.block = block
        JinjaCppFile.__init__(self, {})
        self.ast_dict.update({
            'block': block,
            'config_block': config_block,
            'parameters': self.parameter_definitions
        })

    @property
    def symbols_defined(self):
        return self.block.undefined_symbols

    def __repr__(self):
        self.ast_dict.parameters = self.parameter_definitions
        return super().__repr__()

    def __str__(self):
        self.ast_dict.parameters = self.parameter_definitions
        return super().__str__()

    @property
    def parameter_definitions(self):
        parameter_definitions = [SympyAssignment(s, GetParameter(self.ast_dict.config_block, s))
                                 for s in self.block.undefined_symbols]
        return parameter_definitions

    @property
    def config_required(self):
        return {self.ast_dict.config_block, {s.name: None for s in self.symbols_defined}}


class FieldCopyFunctor(JinjaCppFile):

    TEMPLATE = jinja2.Template("""cuda::fieldCpyFunctor<{{from_type}}, {{ to_type }} >({{from_id}}, {{to_id}})""")  # noqa

    def __init__(self, from_id, from_type, to_id, to_type):
        ast_dict = {'from_id': from_id,
                    'from_type': from_type,
                    'to_id': to_id,
                    'to_type': to_type
                    }
        super().__init__(ast_dict)

    headers = ['"cuda/FieldCopy.h"']


class DefineKernelObjects(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
// Kernels
{% for k in kernels -%}
{{ k }}
{% endfor %}
// Execution
{{ block  }}
""")  # noqa

    def __init__(self, block):
        self.sweeps = block.atoms(SweepOverAllBlocks)
        self.kernels = [SympyAssignment(k.ast_dict.functor.symbol, k.ast_dict.functor,
                                        is_const=False, use_auto=True) for k in self.sweeps]
        ast_dict = {'block': block,
                    'kernels': self.kernels,
                    }
        JinjaCppFile.__init__(self, ast_dict)


class AllocateAllFields(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
// CPU Arrays
{% for c in cpu_arrays %}
{{- c }}
{% endfor %}
// GPU Arrays
{% for g in gpu_arrays %}
{{- g }}
{% endfor %}
{{ block  }}
""")  # noqa

    def __init__(self, block_forest, data_handling, lb_index_shape=(-1,), lb_model_name=None):
        self.data_handling = data_handling
        self.block_forest = block_forest
        self.lb_index_shape = lb_index_shape
        self.lb_model_name = lb_model_name
        JinjaCppFile.__init__(self, {})
        self._update()

    def __repr__(self):
        self._update()
        return super().__repr__()

    def __str__(self):
        self._update()
        return super().__str__()

    def _update(self):
        self.ast_dict.update({
            'cpu_arrays': self._cpu_allocations.values(),
            'gpu_arrays': self._gpu_allocations.values()
        })

    @property
    def _cpu_allocations(self):
        return {s: FieldAllocation(self.block_forest, self.data_handling.fields[s])
                for s in self.data_handling.cpu_arrays.keys()}
        # if not (self.data_handling.fields[s].index_shape == self.lb_index_shape)}

    @property
    def _gpu_allocations(self):
        return {s: FieldAllocation(self.block_forest, self.data_handling.fields[s], on_gpu=True)
                for s in self.data_handling.gpu_arrays.keys()}
        # if not (self.data_handling.fields[s].index_shape == self.lb_index_shape)}


class InitBoundaryHandling(JinjaCppFile):
    TEMPLATE = jinja2.Template("""
// This is my fluid 🏊. It's special... 🤔 but for the computer just a 0
{{ fluid_uid_definition }}

// Initialize geometry 🎲
{{ geometry_initialization }}

// Here are the generated boundaries. They are not so special... 👎
{%- for b in generated_boundaries %}
{{ b }}
{% endfor %}
""")  # noqa

    def __init__(self,
                 block_forest,
                 flag_field_id,
                 pdf_field_id,
                 boundary_conditions,
                 boundary_kernel: dict,
                 field_allocations):
        self.fluid = FlagUidDefinition("fluid")
        ast_dict = {'fluid_uid_definition': self.fluid,
                    'geometry_initialization': BoundaryHandlingFromConfig(block_forest,
                                                                          flag_field_id,
                                                                          self.fluid.symbol),
                    'generated_boundaries': [GeneratedBoundaryInitialization(block_forest,
                                                                             b,
                                                                             pdf_field_id,
                                                                             flag_field_id,
                                                                             self.fluid.symbol,
                                                                             boundary_kernel[b.name],
                                                                             field_allocations)
                                             for b in boundary_conditions]
                    }
        super().__init__(ast_dict)

    headers = ['"cuda/FieldCopy.h"', '"geometry/InitBoundaryHandling.h"']
    @property
    def undefined_symbols(self):
        rtn = super().undefined_symbols
        for b in self.ast_dict.generated_boundaries:
            rtn = rtn | b.undefined_symbols
        return rtn


class GeneratedBoundaryInitialization(JinjaCppFile):
    TEMPLATE = jinja2.Template("""lbm::{{ boundary_condition }} {{ identifier }}( {{ block_forest }}, {{ pdf_field_id }}{{ parameter_str }} );
{{ identifier }}.fillFromFlagField<FlagField_T>( {{ block_forest }}, {{ flag_field_id }}, FlagUID("{{ boundary_condition }}"), {{ fluid_uid }} );
""")  # noqa

    def __init__(self,
                 block_forest,
                 boundary_condition,
                 pdf_field_id,
                 flag_field_id,
                 fluid_uid,
                 kernel,
                 field_allocations):
        def resolve_parameter(p):
            if kernel.target == 'cpu':
                dict = field_allocations._cpu_allocations
            else:
                dict = field_allocations._gpu_allocations

            return dict.get(p.symbol.name.replace('_data_', ''), p).symbol
        parameters = kernel.get_parameters()
        parameter_ids = [resolve_parameter(p)
                         for p in parameters
                         if (p.is_field_pointer or not p.is_field_parameter) and
                         p.symbol.name not in ('_data_indexVector', '_data_pdfs', 'indexVectorSize',)]
        parameter_str = ', '.join(p.name for p in parameter_ids)
        if parameter_str:
            parameter_str = ', ' + parameter_str

        self.fluid = FlagUidDefinition("fluid")
        ast_dict = {'block_forest': block_forest,
                    'boundary_condition': pascalcase(boundary_condition.name),
                    'identifier': camelcase(boundary_condition.name),
                    'pdf_field_id': pdf_field_id,
                    'fluid_uid': fluid_uid,
                    'flag_field_id': flag_field_id,
                    'parameter_ids': parameter_ids,
                    'parameter_str': parameter_str
                    }
        super().__init__(ast_dict)

    @property
    def headers(self):
        return [f'"{pascalcase(self.ast_dict.boundary_condition)}.h"']

    @property
    def symbols_defined(self):
        # Could also be defined
        # TypedSymbol(self.ast_dict.identifier, 'FlagUID'),
        return {TypedSymbol(self.ast_dict.identifier, f'lbm::{self.ast_dict.boundary_condition}')}

    @property
    def undefined_symbols(self):
        return super().undefined_symbols | set(self.ast_dict.parameter_ids)


class SweepCreation(JinjaCppFile):
    TEMPLATE = jinja2.Template("""{{ sweep_class_name }}( {{ parameter_str }} )""")  # noqa

    def __init__(self,
                 sweep_class_name: str,
                 field_allocation: AllocateAllFields,
                 ast,
                 parameters_to_ignore=[]):
        def resolve_parameter(p):
            if ast.target == 'cpu':
                dict = field_allocation._cpu_allocations
            else:
                dict = field_allocation._gpu_allocations

            return dict.get(p.symbol.name.replace('_data_', ''), p).symbol

        parameters = ast.get_parameters()
        parameter_ids = [resolve_parameter(p)
                         for p in parameters
                         if (p.is_field_pointer and p.symbol.field_name not in parameters_to_ignore)
                         or not p.is_field_parameter]

        ast_dict = {'sweep_class_name': sweep_class_name,
                    'parameter_ids': parameter_ids,
                    'parameter_str': ', '.join(p.name for p in parameter_ids)}
        super().__init__(ast_dict)
        self.symbol = TypedSymbol(self.ast_dict.sweep_class_name.lower(), self.ast_dict.sweep_class_name)

    @property
    def headers(self):
        return [f'"{self.ast_dict.sweep_class_name}.h"']

    @property
    def undefined_symbols(self):
        return set(self.ast_dict.parameter_ids)

    @property
    def symbols_defined(self):
        return {self.symbol}

    def __sympy__(self):
        return self.symbol


class SweepOverAllBlocks(JinjaCppFile):
    # TEMPLATE = jinja2.Template("""std::for_each({{block_forest}}->begin(), {{block_forest}}->end(), {{functor}});""")  # noqa
    TEMPLATE = jinja2.Template("""sweep({{block_forest}}, {{sweep_class_name | lower() }});""")

    def __init__(self, functor: SweepCreation, block_forest):
        ast_dict = {'functor': functor,
                    'sweep_class_name': functor.ast_dict.sweep_class_name,
                    'block_forest': block_forest}
        super().__init__(ast_dict)

    @property
    def symbols_undefined(self):
        return {self.ast_dict.functor.symbol}

    @property
    def required_global_declarations(self):
        return ["""template < class BlockStorage_T, class Functor_T >
static inline auto sweep(walberla::shared_ptr<BlockStorage_T> blocks, Functor_T functor) -> void {
   for ( auto& block : *blocks ) {
      functor(&block);
   }
}
"""]


class FieldCopy(JinjaCppFile):
    TEMPLATE = jinja2.Template("""cuda::fieldCpy < {{src_type }}, {{dst_type }} > ({{block_forest }}, {{src_id }}, {{dst_id }}); """)  # noqa

    def __init__(self, block_forest, src_id, src_field, src_gpu, dst_id, dst_field, dst_gpu):
        src_type = _make_field_type(src_field, src_gpu)
        dst_type = _make_field_type(dst_field, dst_gpu)
        ast_dict = {'src_id': src_id,
                    'dst_id': dst_id,
                    'src_type': src_type,
                    'dst_type': dst_type,
                    'block_forest': block_forest}
        super().__init__(ast_dict)

    headers = ['"cuda/FieldCopy.h"']


class SwapFields(JinjaCppFile):
    """
     .. warn::

         Prefer temporary fields in sweeps over this class! Two full fields have higher memory usage.

    """
    TEMPLATE = jinja2.Template("""std::swap({{field_id0}}, {{field_id1}});""")

    def __init__(self, field_id0, field_id1):
        ast_dict = {'field_id0': field_id0,
                    'field_id1': field_id1}
        super().__init__(ast_dict)

    headers = ["<algorithm>"]


class Communication(JinjaCppFile):
    """
     .. warn::

         Prefer temporary fields in sweeps over this class! Two full fields have higher memory usage.

    """
    TEMPLATE = jinja2.Template("""communication();""")

    def __init__(self, gpu):
        ast_dict = {'gpu': gpu}
        super().__init__(ast_dict)

    headers = ["<algorithm>"]
    headers = ["<algorithm>"]
