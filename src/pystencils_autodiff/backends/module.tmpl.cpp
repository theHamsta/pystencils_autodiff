#define RESTRICT __restrict__

//#if GOOGLE_CUDA
//#define EIGEN_USE_GPU
//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#endif

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
