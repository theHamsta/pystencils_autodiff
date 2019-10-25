# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""
Astnodes useful for the generations of C modules for frameworks (apart from waLBerla)

waLBerla currently uses `pystencils-walberla <https://pypi.org/project/pystencils-walberla/>`_.
"""
import itertools
from collections.abc import Iterable
from functools import reduce
from typing import Any, List, Set

import jinja2

import pystencils
import sympy as sp
from pystencils.astnodes import KernelFunction, Node, NodeOrExpr, ResolvedFieldAccess
from pystencils.data_types import TypedSymbol
from pystencils.kernelparameters import FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol
from pystencils_autodiff.framework_integration.printer import FrameworkIntegrationPrinter


class DestructuringBindingsForFieldClass(Node):
    """
    Defines all variables needed for describing a field (shape, pointer, strides)
    """
    headers = ['<PyStencilsField.h>']
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "data<{dtype}>",
        FieldShapeSymbol: "shape[{dim}]",
        FieldStrideSymbol: "stride[{dim}]"
    }
    CLASS_NAME_TEMPLATE = "PyStencilsField<{dtype}, {ndim}>"

    @property
    def fields_accessed(self) -> Set['ResolvedFieldAccess']:
        """Set of Field instances: fields which are accessed inside this kernel function"""

        # TODO: remove when texture support is merged into pystencils
        try:
            from pystencils.interpolation_astnodes import TextureAccess
            return set(o.field for o in self.atoms(ResolvedFieldAccess) | self.atoms(TextureAccess))
        except ImportError:
            return set(o.field for o in self.atoms(ResolvedFieldAccess))

    def __init__(self, body):
        super(DestructuringBindingsForFieldClass, self).__init__()
        self.body = body
        body.parent = self

    @property
    def args(self) -> List[NodeOrExpr]:
        """Returns all arguments/children of this node."""
        return [self.body]

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        """Set of symbols which are defined by this node."""
        undefined_field_symbols = {s for s in self.body.undefined_symbols
                                   if isinstance(s, (FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol))}
        return undefined_field_symbols

    @property
    def undefined_symbols(self) -> Set[sp.Symbol]:
        field_map = {f.name: f for f in self.fields_accessed}
        undefined_field_symbols = self.symbols_defined
        corresponding_field_names = {s.field_name for s in undefined_field_symbols if hasattr(s, 'field_name')}
        corresponding_field_names |= {s.field_names[0] for s in undefined_field_symbols if hasattr(s, 'field_names')}
        return {TypedSymbol(f, self.CLASS_NAME_TEMPLATE.format(dtype=field_map[f].dtype, ndim=field_map[f].ndim) + '&')
                for f in corresponding_field_names} | (self.body.undefined_symbols - undefined_field_symbols)

    def subs(self, subs_dict) -> None:
        """Inplace! substitute, similar to sympy's but modifies the AST inplace."""
        self.body.subs(subs_dict)

    def __repr__(self):
        return f'Destructuring of Tensors {self.symbols_defined}\n' + self.body.__repr__()

    @property
    def func(self):
        return self.__class__

    def atoms(self, arg_type) -> Set[Any]:
        return self.body.atoms(arg_type) | {s for s in self.symbols_defined if isinstance(s, arg_type)}


class NativeTextureBinding(pystencils.backends.cbackend.CustomCodeNode):
    """
    Bind texture to CUDA device pointer

    Recommended read: https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/

    The definition from cudaResourceDesc and cudaTextureDesc

    .. code:: c

        /**
         * CUDA resource descriptor
         */
        struct __device_builtin__ cudaResourceDesc {
            enum cudaResourceType resType;             /**< Resource type */

            union {
                struct {
                    cudaArray_t array;                 /**< CUDA array */
                } array;
                struct {
                    cudaMipmappedArray_t mipmap;       /**< CUDA mipmapped array */
                } mipmap;
                struct {
                    void *devPtr;                      /**< Device pointer */
                    struct cudaChannelFormatDesc desc; /**< Channel descriptor */
                    size_t sizeInBytes;                /**< Size in bytes */
                } linear;
                struct {
                    void *devPtr;                      /**< Device pointer */
                    struct cudaChannelFormatDesc desc; /**< Channel descriptor */
                    size_t width;                      /**< Width of the array in elements */
                    size_t height;                     /**< Height of the array in elements */
                    size_t pitchInBytes;               /**< Pitch between two rows in bytes */
                } pitch2D;
            } res;
        };

    .. code:: c

        /**
         * CUDA texture descriptor
         */
        struct __device_builtin__ cudaTextureDesc
        {
            /**
             * Texture address mode for up to 3 dimensions
             */
            enum cudaTextureAddressMode addressMode[3];
            /**
             * Texture filter mode
             */
            enum cudaTextureFilterMode  filterMode;
            /**
             * Texture read mode
             */
            enum cudaTextureReadMode    readMode;
            /**
             * Perform sRGB->linear conversion during texture read
             */
            int                         sRGB;
            /**
             * Texture Border Color
             */
            float                       borderColor[4];
            /**
             * Indicates whether texture reads are normalized or not
             */
            int                         normalizedCoords;
            /**
             * Limit to the anisotropy ratio
             */
            unsigned int                maxAnisotropy;
            /**
             * Mipmap filter mode
             */
            enum cudaTextureFilterMode  mipmapFilterMode;
            /**
             * Offset applied to the supplied mipmap level
             */
            float                       mipmapLevelBias;
            /**
             * Lower end of the mipmap level range to clamp access to
             */
            float                       minMipmapLevelClamp;
            /**
             * Upper end of the mipmap level range to clamp access to
             */
            float                       maxMipmapLevelClamp;
        };
    """  # noqa
    CODE_TEMPLATE_LINEAR = jinja2.Template("""cudaResourceDesc {{resource_desc}}{};
{{resource_desc}}.resType = cudaResourceTypeLinear;
{{resource_desc}}.res.linear.devPtr = {{device_ptr}};
{{resource_desc}}.res.linear.desc.f = {{cuda_channel_format}};
{{resource_desc}}.res.linear.desc.x = {{bits_per_channel}}; // bits per channel
{{resource_desc}}.res.linear.sizeInBytes = {{total_size}};

cudaTextureDesc {{texture_desc}}{};
cudaTextureObject_t {{texture_object}}=0;
cudaCreateTextureObject(&{{texture_object}}, &{{resource_desc}}, &{{texture_desc}}, nullptr);
{{texture_desc}}.readMode = cudaReadModeElementType;
auto {{texture_object}}Destroyer = std::unique_ptr(nullptr, [&](){
   cudaDestroyTextureObject({{texture_object}});
});
    """)
    CODE_TEMPLATE_PITCHED2D = jinja2.Template(""" !!! TODO!!! """)
    CODE_TEMPLATE_CUDA_ARRAY = jinja2.Template("""
auto channel_desc_{{texture_name}} = {{channel_desc}};
{{ create_array }}
{{ copy_array }}
{{ texture_name }}.addressMode[0] = cudaAddressModeBorder;
{{ texture_name }}.addressMode[1] = cudaAddressModeBorder;
{{ texture_name }}.addressMode[2] = cudaAddressModeBorder;
{{ texture_name }}.filterMode = cudaFilterModeLinear;
{{ texture_name }}.normalized = false;
cudaBindTextureToArray(&{{texture_name}}, {{array}}, &channel_desc_{{texture_name}});
std::shared_ptr<void> {{array}}Destroyer(nullptr, [&](...){
    cudaFreeArray({{array}});
    cudaUnbindTexture({{texture_name}});
});
    """)

    def __init__(self, texture, device_data_ptr, use_texture_objects=True):
        self._texture = texture
        self._device_ptr = device_data_ptr
        self._dtype = self._device_ptr.dtype.base_type
        self._shape = tuple(sp.S(s) for s in self._texture.field.shape)
        self._ndim = texture.field.ndim
        assert use_texture_objects, "without texture objects is not implemented"

        super().__init__(self.get_code(dialect='c', vector_instruction_set=None),
                         symbols_read={device_data_ptr,
                                       *[s for s in self._shape if isinstance(s, sp.Symbol)]},
                         symbols_defined={})
        self.headers = ['<memory>', '<cuda.h>', '<cuda_runtime_api.h>']

    def get_code(self, dialect, vector_instruction_set):
        texture_name = self._texture.symbol.name
        code = self.CODE_TEMPLATE_CUDA_ARRAY.render(
            resource_desc='resDesc_' + texture_name,
            texture_desc='texDesc_' + texture_name,
            channel_desc=f'cudaCreateChannelDesc<{self._dtype}>()',  # noqa
            texture_object='tex_' + texture_name,
            array='array_' + texture_name,
            texture_name=texture_name,
            ndim=self._ndim,
            device_ptr=self._device_ptr,
            create_array=self._get_create_array_call(),
            copy_array=self._get_copy_array_call(),
            dtype=self._dtype,
            bits_per_channel=self._dtype.numpy_dtype.itemsize * 8,
            total_size=self._dtype.numpy_dtype.itemsize * reduce(lambda x, y: x * y, self._shape, 1))
        return code

    def _get_create_array_call(self):
        texture_name = self._texture.symbol.name
        ndim = '' if self._ndim <= 2 else f'{self._ndim}D'
        array = 'array_' + texture_name
        return f"""cudaArray * {array};
cudaMalloc{ndim}Array(&{array}, &channel_desc_{texture_name}, """ + (
            (f'{{{", ".join(str(s) for s in reversed(self._shape))}}});'
                if self._ndim == 3
                else f'{", ".join(str(s) for s in reversed(self._shape))});'))

    def _get_copy_array_call(self):
        texture_name = self._texture.symbol.name
        array = 'array_' + texture_name
        if self._texture.field.ndim == 3:
            copy_params = f'cpy_{texture_name}_params'
            return f"""cudaMemcpy3DParams {copy_params}{{}};
{copy_params}.srcPtr = {{{self._device_ptr},
                       {self._texture.field.strides[-1] * self._texture.field.shape[-1] * self._dtype.numpy_dtype.itemsize},
                       {self._texture.field.shape[-1], self._texture.field.shape[-2]}}};
{copy_params}.dsrArray = {array};
copyParams.extent = {{{", ".join(reversed(self._shape))}}};
copyParams.kind = cudaMemcpyDeviceToDevice;
cudaMemcpy3D(&{{copy_params}});"""  # noqa
        elif self._texture.field.ndim == 2:
            # noqa: cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);

            return f"""cudaMemcpy2DToArray({array},
                    0u,
                    0u,
                    {self._device_ptr},
                    {self._texture.field.strides[-1] * self._texture.field.shape[-1] * self._dtype.numpy_dtype.itemsize},
                    {self._texture.field.shape[-1]},
                    {self._texture.field.shape[-2]},
                    cudaMemcpyDeviceToDevice);
 """  # noqa
        else:
            raise NotImplementedError()


class KernelFunctionCall(Node):
    """
    AST nodes representing a call of a :class:`pystencils.astnodes.KernelFunction`
    """

    def __init__(self, kernel_function_node: pystencils.astnodes.KernelFunction):
        self.kernel_function = kernel_function_node

    @property
    def args(self):
        return [self.kernel_function]

    @property
    def symbols_defined(self) -> Set[sp.Symbol]:
        return set()

    @property
    def undefined_symbols(self) -> Set[sp.Symbol]:
        return {p.symbol for p in self.kernel_function.get_parameters()}

    def subs(self, subs_dict) -> None:
        for a in self.args:
            a.subs(subs_dict)

    @property
    def func(self):
        return self.__class__

    def __repr__(self):
        return f"call {self.kernel_function.function_name}{self.kernel_function.get_parameters()}"


class WrapperFunction(pystencils.astnodes.KernelFunction):

    def __init__(self, body, function_name='wrapper', target='cpu', backend='c'):
        super().__init__(body, target, backend, compile_function=None, ghost_layers=0)
        self.function_name = function_name


def generate_kernel_call(kernel_function):
    try:
        from pystencils.interpolation_astnodes import TextureAccess
        from pystencils.kernelparameters import FieldPointerSymbol

        textures = {a.texture for a in kernel_function.atoms(TextureAccess)}
        texture_uploads = [NativeTextureBinding(t, FieldPointerSymbol(t.field.name, t.field.dtype, const=True))
                           for t in textures]
    except ImportError:
        texture_uploads = []

    if texture_uploads:
        block = pystencils.astnodes.Block([
            *texture_uploads,
            KernelFunctionCall(kernel_function)
        ])
    else:
        return pystencils.astnodes.Block([KernelFunctionCall(kernel_function)])

    return block


class JinjaCppFile(Node):
    TEMPLATE: jinja2.Template = None

    def __init__(self, ast_dict):
        self.ast_dict = ast_dict
        self.printer = FrameworkIntegrationPrinter()
        Node.__init__(self)

    @property
    def args(self):
        """Returns all arguments/children of this node."""
        ast_nodes = [a for a in self.ast_dict.values() if isinstance(a, (Node, str))]
        iterables_of_ast_nodes = [a for a in self.ast_dict.values() if isinstance(a, Iterable)]
        return ast_nodes + list(itertools.chain.from_iterable(iterables_of_ast_nodes))

    @property
    def symbols_defined(self):
        """Set of symbols which are defined by this node."""
        return set()

    @property
    def undefined_symbols(self):
        """Symbols which are used but are not defined inside this node."""
        return set()

    def _print(self, node):
        if isinstance(node, Node):
            return self.printer(node)
        else:
            return str(node)

    def atoms(self, arg_type) -> Set[Any]:
        """Returns a set of all descendants recursively, which are an instance of the given type."""
        result = set()
        for arg in self.args:
            if isinstance(arg, arg_type):
                result.add(arg)
            if hasattr(arg, 'atoms'):
                result.update(arg.atoms(arg_type))
        return result

    @property
    def is_cuda(self):
        return any(f.backend == 'gpucuda' for f in self.atoms(KernelFunction))

    def __str__(self):
        assert self.TEMPLATE, f"Template of {self.__class__} must be set"
        render_dict = {k: (self._print(v) if not isinstance(v, (pystencils.Field, pystencils.TypedSymbol)) else v)
                       if not isinstance(v, Iterable) or isinstance(v, str)
                       else [(self._print(a)
                              if not isinstance(a, (pystencils.Field, pystencils.TypedSymbol))
                              else a)
                             for a in v]
                       for k, v in self.ast_dict.items()}
        render_dict.update({"headers": pystencils.backends.cbackend.get_headers(self)})
        render_dict.update({"globals": pystencils.backends.cbackend.get_global_declarations(self)})

        return self.TEMPLATE.render(render_dict)

    def __repr__(self):
        return f'{str(self.__class__)}:\n {self.TEMPLATE.render(self.ast_dict)}'
