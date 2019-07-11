#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
using namespace pybind11::literals;

void {{ kernel_name }}_cuda_forward(
{%- for tensor in forward_tensors %}
    at::Tensor {{ tensor.name }} {{- ", " if not loop.last -}}
{% endfor %});

std::vector<at::Tensor> {{ kernel_name }}_cuda_backward(
{%- for tensor in backward_tensors -%}
    at::Tensor {{ tensor.name }} {{- ", " if not loop.last -}}
{% endfor %});

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  //AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x);       
  //CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> {{ kernel_name }}_forward(
{%- for tensor in forward_tensors -%}
    at::Tensor {{ tensor.name }} {{- ", " if not loop.last -}}
{%- endfor %})
{
    {% for tensor in forward_tensors -%}
    CHECK_INPUT({{ tensor.name }});
    {% endfor %}

    {{ kernel_name }}_cuda_forward(
        {%- for tensor in forward_tensors %}
        {{ tensor.name }} {{- ", " if not loop.last }}
        {%- endfor %});

    return std::vector<at::Tensor>{
        {%- for tensor in forward_output_tensors %}
        {{ tensor.name }} {{- ", " if not loop.last }}
        {%- endfor %}
    }
        ;
}

std::vector<at::Tensor> {{ kernel_name }}_backward(
{%- for tensor in backward_tensors -%}
    at::Tensor {{ tensor.name }} {{- ", " if not loop.last -}}
{% endfor %})
{
    {%- for tensor in forward_input_tensors + backward_input_tensors -%}
    CHECK_INPUT({{ tensor }});
    {% endfor %}
    {{ kernel_name }}_cuda_backward(
        {%- for tensor in backward_tensors -%}
        {{ tensor.name }} {{- ", " if not loop.last }}
        {%- endfor %});

    return std::vector<at::Tensor>{
        {%- for tensor in backward_output_tensors %}
        {{ tensor.name }} {{- ", " if not loop.last }}
        {%- endfor %}
    }
        ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &{{ kernel_name }}_forward, "{{ kernel_name }} forward (CUDA)",
{%- for tensor in forward_tensors -%}
    "{{ tensor.name }}"_a {{ ", " if not loop.last }}  
{%- endfor -%} );
  m.def("backward", &{{ kernel_name }}_backward, "{{ kernel_name }} backward (CUDA)",
{%- for tensor in backward_tensors -%}
    "{{ tensor.name }}"_a {{ ", " if not loop.last }}  
{%- endfor -%} );
}
