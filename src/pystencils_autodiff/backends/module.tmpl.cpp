#ifdef _MSC_BUILD
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

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
