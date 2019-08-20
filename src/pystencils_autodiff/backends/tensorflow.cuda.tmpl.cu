#define RESTRICT __restrict__

#define GOOGLE_CUDA 1

#define EIGEN_USE_GPU 1

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
