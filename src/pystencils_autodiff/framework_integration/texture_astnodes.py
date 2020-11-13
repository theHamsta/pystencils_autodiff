# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from functools import reduce

import jinja2
import sympy as sp

import pystencils.backends
import pystencils.kernelparameters


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
    CODE_TEMPLATE_LINEAR = jinja2.Template("""
cudaResourceDesc {{resource_desc}}{};
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
//#   pragma GCC diagnostic ignored "-Wconversion"
auto channel_desc_{{texture_name}} = {{channel_desc}};
{{ create_array }}
{{ copy_array }}
    cudaDeviceSynchronize();
{{texture_namespace}}{{ texture_name }}.addressMode[0] = cudaAddressModeBorder;
{{texture_namespace}}{{ texture_name }}.addressMode[1] = cudaAddressModeBorder;
{{texture_namespace}}{{ texture_name }}.addressMode[2] = cudaAddressModeBorder;
{{texture_namespace}}{{ texture_name }}.filterMode = cudaFilterModeLinear;
{{texture_namespace}}{{ texture_name }}.normalized = false;
cudaBindTextureToArray(&{{texture_namespace}}{{texture_name}}, {{array}}, &channel_desc_{{texture_name}});
std::shared_ptr<void> {{array}}Destroyer(nullptr, [&](...){
    cudaDeviceSynchronize();
    cudaFreeArray({{array}});
    cudaUnbindTexture({{texture_namespace}}{{texture_name}});
});
// #pragma GCC diagnostic pop
""")

    def __init__(self, texture, device_data_ptr, use_texture_objects=True, texture_namespace=''):
        self._texture = texture
        self._device_ptr = device_data_ptr
        self._dtype = self._device_ptr.dtype.base_type
        self._shape = tuple(sp.S(s) for s in self._texture.field.shape)
        self._ndim = texture.field.ndim
        self._texture_namespace = texture_namespace
        assert use_texture_objects, "without texture objects is not implemented"

        super().__init__(self.get_code(dialect='c', vector_instruction_set=None),
                         symbols_read={device_data_ptr,
                                       *[s for s in self._shape if isinstance(s, sp.Symbol)]},
                         symbols_defined={})
        self.headers = ['<memory>', '<cuda.h>', '<cuda_runtime_api.h>']

    def get_code(self, dialect='', vector_instruction_set=''):
        texture_name = self._texture.symbol.name
        code = self.CODE_TEMPLATE_CUDA_ARRAY.render(
            resource_desc='resDesc_' + texture_name,
            texture_desc='texDesc_' + texture_name,
            channel_desc=f'cudaCreateChannelDesc<{self._dtype}>()',  # noqa
            texture_object='tex_' + texture_name,
            array='array_' + texture_name,
            texture_name=texture_name,
            texture_namespace=self._texture_namespace + '::' if self._texture_namespace else '',
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
            return f"""cudaMemcpy3DParms {copy_params}{{}};
{copy_params}.srcPtr = {{{self._device_ptr},
                        {self._texture.field.strides[-2] * self._texture.field.dtype.numpy_dtype.itemsize},
                        {self._texture.field.shape[-1]},
                        {self._texture.field.shape[-2]}}};
{copy_params}.dstArray = {array};
{copy_params}.extent = {{{", ".join(str(s) for s in reversed(self._shape))}}};
{copy_params}.kind = cudaMemcpyDeviceToDevice;
cudaMemcpy3D(&{copy_params});"""  # noqa
        elif self._texture.field.ndim == 2:
            # noqa, cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
            return f"""cudaMemcpy2DToArray({array},
                    0u,
                    0u,
                    {self._device_ptr},
                    {self._texture.field.strides[-2] * self._texture.field.dtype.numpy_dtype.itemsize},
                    {self._texture.field.shape[-1] * self._texture.field.dtype.numpy_dtype.itemsize}, // Dafaq, this has to be in bytes, but only columns and only in memcpy2D
                    {self._texture.field.shape[-2]},
                    cudaMemcpyDeviceToDevice);
 """  # noqa
        else:
            raise NotImplementedError()

    @property
    def undefined_symbols(self):
        field = self._texture.field
        return sp.S(field.strides[-2]).free_symbols \
            | sp.S(sp.Add(*field.shape[-field.ndim:])).free_symbols \
            | {pystencils.kernelparameters.FieldPointerSymbol(field.name, field.dtype, const=True)}
