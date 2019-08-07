#include <torch/extension.h>

#include <vector>

using namespace pybind11::literals;

using scalar_t = {{ dtype }};


#define RESTRICT __restrict

std::vector<at::Tensor> {{ kernel_name }}_forward(
{%- for tensor in forward_tensors -%}
    at::Tensor {{ tensor }} {{- ", " if not loop.last -}}
{%- endfor %})
{
    //{% for tensor in forward_output_tensors -%}
    //auto {{tensor}} = at::zeros_like({{ forward_input_tensors[0] }});
    //{% endfor %}

    {% for tensor in forward_tensors -%}
    {%- set last = loop.last -%}
    scalar_t* _data_{{ tensor }} = {{ tensor }}.data<scalar_t>();
    {% for i in dimensions -%}
    int _stride_{{tensor}}_{{i}} = {{tensor}}.strides()[{{ i }}];
    {% endfor -%}
    {% for i in dimensions -%}
    int _size_{{tensor}}_{{i}} = {{tensor}}.size({{ i }});
    {% endfor -%}
    {% endfor -%}

    {{forward_kernel}}

    return {
    {%- for tensor in forward_output_tensors -%}
    {{ tensor }} {{- "," if not loop.last -}}
    {% endfor -%}
    };
}

std::vector<at::Tensor> {{ kernel_name }}_backward(
{%- for tensor in backward_tensors -%}
    at::Tensor {{ tensor }} {{- ", " if not loop.last -}}
{% endfor %})
{
    //{% for tensor in backward_output_tensors -%}
    //auto {{tensor}} = at::zeros_like({{ backward_input_tensors[0] }});
    //{% endfor %}

    {% for tensor in backward_tensors -%}
    {%- set last = loop.last -%}
    scalar_t* _data_{{ tensor }} = {{ tensor }}.data<scalar_t>();
    {% for i in dimensions -%}
    int _stride_{{ tensor }}_{{i}} = {{ tensor }}.strides()[{{ i }}];
    {% endfor -%}
    {% for i in dimensions -%}
    int _size_{{tensor}}_{{i}} = {{tensor}}.size({{ i }});
    {% endfor -%}
    {% endfor -%}

    {{backward_kernel}}

    return {
    {%- for tensor in backward_output_tensors -%}
    {{ tensor }} {{- "," if not loop.last -}}
    {% endfor -%}
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &{{ kernel_name }}_forward, "{{ kernel_name }} forward (CPU)",
{%- for tensor in forward_tensors -%}
    "{{ tensor }}"_a {{ ", " if not loop.last }}  
{%- endfor -%} );
  m.def("backward", &{{ kernel_name }}_backward, "{{ kernel_name }} backward (CPU)",
{%- for tensor in backward_tensors -%}
    "{{ tensor }}"_a {{ ", " if not loop.last }}  
{%- endfor -%} );
}
