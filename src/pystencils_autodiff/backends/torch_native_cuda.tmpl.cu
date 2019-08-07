
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define RESTRICT __restrict__

{% for g in cuda_globals -%}
{{ g }}
{% endfor %}

template <typename scalar_t>
__global__ void {{ kernel_name }}_cuda_forward_kernel(
        {% for tensor in forward_tensors -%}
        {%- set last = loop.last -%}
        scalar_t* __restrict__ _data_{{ tensor.name }},
        {% for i in range(tensor.spatial_dimensions )-%}
        int _stride_{{ tensor.name }}_{{ i }} {{- ", " }} 
        {% endfor -%} 
        {% endfor -%}
        {% for i in range(forward_output_tensors[0].spatial_dimensions )-%}
        int _size_{{ forward_output_tensors[0] }}_{{ i }} {{- "," if not loop.last }}
        {% endfor %})
{
    {{forward_kernel}}
}
    
template <typename scalar_t>
__global__ void {{ kernel_name }}_cuda_backward_kernel(
        {% for tensor in backward_tensors -%}
        {%- set last = loop.last -%}
        scalar_t* __restrict__ _data_{{ tensor.name }},
        {% for i in range(tensor.spatial_dimensions )-%}
        int _stride_{{ tensor.name }}_{{ i }} {{- ", " }}
        {% endfor -%}
        {% endfor -%}
        {% for i in range(forward_output_tensors[0].spatial_dimensions )-%}
        int _size_{{ forward_output_tensors[0].name }}_{{ i }} {{- "," if not loop.last }}
        {% endfor %})
{
    {{backward_kernel}}
}

void {{ kernel_name }}_cuda_forward(
    {%- for tensor in forward_tensors -%}
    at::Tensor {{ tensor.name }} {{- "," if not loop.last -}}
    {%- endfor -%})
{

    {% for i in range(forward_output_tensors[0].spatial_dimensions )-%}
    int _size_{{ forward_output_tensors[0].name }}_{{ i }} = {{ forward_output_tensors[0].name }}.size({{ i }});
    {% endfor %}

/*at:: at::device(at::kCUDA).dtype(k{{ dtype }})*/
    AT_DISPATCH_FLOATING_TYPES({{ forward_input_tensors[0].name }}.type(), "{{ kernel_name }}_forward_cuda", ([&] {
                {{ kernel_name }}_cuda_forward_kernel<scalar_t><<<dim3{{ forward_blocks }}, dim3{{ forward_threads }}>>>(
                        {% for tensor in forward_tensors -%}
                        {%- set last = loop.last -%}
                        {{tensor.name}}.data<scalar_t>(),
                        {% for i in range(tensor.spatial_dimensions) -%}
                        {{tensor.name}}.strides()[{{ i }}] {{- "," }}
                        {% endfor -%}
                        {% endfor -%}
                        {% for i in range(forward_output_tensors[0].spatial_dimensions) -%}
                        {{ forward_output_tensors[0].name }}.size({{ i }}) {{- "," if not loop.last }}
                        {% endfor %}
                        );
                }));
     cudaError_t err = cudaGetLastError();
     if (err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        throw err;
     }
}

void {{ kernel_name }}_cuda_backward(
    {%- for tensor in backward_tensors -%}
    at::Tensor {{ tensor.name }} {{- ", " if not loop.last -}}
    {%- endfor %})
{

    {% for i in range(backward_output_tensors[0].spatial_dimensions )-%}
    int _size_{{ backward_output_tensors[0].name }}_{{ i }} = {{ backward_output_tensors[0].name }}.size({{ i }});
    {% endfor %}

/*at:: at::device(at::kCUDA).dtype(k{{ dtype }})*/
    AT_DISPATCH_FLOATING_TYPES({{ backward_input_tensors[0].name }}.type(), "{{ kernel_name }}_backward_cuda", ([&] {
                {{ kernel_name }}_cuda_backward_kernel<scalar_t><<<dim3{{ backward_blocks }}, dim3{{ backward_threads }}>>>(
                        {% for tensor in backward_tensors -%}
                        {%- set last = loop.last -%}
                        {{tensor.name}}.data<scalar_t>(),
                        {% for i in range(tensor.spatial_dimensions )-%}
                        {{tensor.name}}.strides()[{{ i }}]{{- ", " }}
                        {% endfor -%}
                        {% endfor -%}
                        {% for i in range(backward_output_tensors[0].spatial_dimensions )-%}
                        {{ backward_output_tensors[0].name }}.size({{ i }}) {{- "," if not loop.last }}
                        {% endfor %}
                        );
                }));

}
