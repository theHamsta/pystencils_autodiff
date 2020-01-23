#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


from os.path import dirname, join

import jinja2

from pystencils.data_types import TypedSymbol
from pystencils_autodiff._file_io import read_template_from_file
from pystencils_autodiff.framework_integration.astnodes import JinjaCppFile


class WalberlaModule(JinjaCppFile):
    TEMPLATE = read_template_from_file(join(dirname(__file__), 'walberla_main.tmpl.cpp'))

    def __init__(self, main):
        ast_dict = {'main': main}
        JinjaCppFile.__init__(self, ast_dict)


class WalberlaMain(JinjaCppFile):

    TEMPLATE = jinja2.Template("""
int main( int argc, char ** argv )
{
   using namespace walberla;
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


class BlockForrestCreation(JinjaCppFile):
    TEMPLATE = jinja2.Template("""auto {{ blocks }} = walberla_user::createBlockForrest(walberlaEnv);""")
    def __init__(self, block_forrest_name):

        ast_dict = {
            'blocks': TypedSymbol(block_forrest_name, "auto")
        }

        super().__init__(ast_dict)

    @property
    def symbols_defined(self):
        return {self.ast_dict.blocks}


class UniformBlockForrestFromConfig(BlockForrestCreation):
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

    def __init__(self, walberla_module):
        self.main_module = walberla_module
        super().__init__({})

    @property
    def config_required(self):
        return {"DomainSetup": {"blocks": [1, 1, 1], "cellsPerBlock": [100, 100, 100]}}


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
        'walberlaEnv.config()->getOneBlock("{{ block }}").getParameter<{{ key.dtype }}>("{{ key }}"{% if default %}, {{ default }}{% endif %})'  # noqa
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

    TEMPLATE = jinja2.Template("""BlockDataID {{ field_name }}_data = field::addToStorage<{{ field_type }}>( {{ block_forrest }},
                              {{ field_name }}
     {%- if init_value -%}      , {{ init_value }} {% endif %}
     {%- if layout_str -%}      , {{ layout_str }} {% endif %}
     {%- if num_ghost_layers -%}, {{ num_ghost_layers }} {% endif %}
     {%- if always_initialize -%}, {{ always_initialize }} {% endif %})
        """)  # noqa

    def __init__(self, block_forrest, field):
        self._symbol = TypedSymbol(field.name + '_data', 'BlockDataID')
        ast_dict = {
            'block_forrest': block_forrest,
            'field_name': field.name,
            'field_type': f'GhostLayerField< {field.dtype}, {field.index_shape[0] if field.index_shape else 1} >'
        }
        super().__init__(ast_dict)

    headers = ['"field/GhostLayerField.h"']

    @property
    def symbol(self):
        return self._symbol

    @property
    def symbols_defined(self):
        return {self.symbol}
