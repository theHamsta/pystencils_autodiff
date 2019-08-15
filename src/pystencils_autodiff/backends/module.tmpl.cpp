#include <cuda.h>
#include <vector>

// Most compilers don't care whether it's __restrict or __restrict__
#define RESTRICT __restrict__

{% for header in headers -%}
#include {{ header }}
{% endfor %}

{% for global in globals -%}
{{ global }}
{% endfor %}

{% for kernel in kernels %}
{{ kernel }}
{% endfor %}

{% for wrapper in kernel_wrappers %}
{{ wrapper }}
{% endfor %}

{{ python_bindings }}

