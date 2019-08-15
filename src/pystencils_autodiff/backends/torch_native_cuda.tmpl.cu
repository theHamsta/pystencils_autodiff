#include <cuda.h>
#include <vector>


{% for header in headers -%}
#include {{ header }}
{% endfor %}

#define RESTRICT __restrict__
#define FUNC_PREFIX __global__


{{ forward_kernel }}

{{ backward_kernel }}


#define RESTRICT
{{ forward_wrapper }}

{{ backward_wrapper }}
