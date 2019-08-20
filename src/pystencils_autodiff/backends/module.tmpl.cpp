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

#include <cuda.h>
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
{{ python_bindings }}
