/* 
 * Automatically generated waLBerla main
 *
 */

{% for header in headers -%}
#include {{ header }}
{% endfor %}
{% for global in globals -%}
{{ global }}
{% endfor %}

{{ main }}
