/*
 * {{ module_name }}.hpp
 * Copyright (C) 2020 Stephan Seitz <stephan.seitz@fau.de>
 *
 * Distributed under terms of the GPLv3 license.
 */

#pragma once

{% for header in headers -%}
#include {{ header }}
{% endfor %}

{{ declarations | join('\n\n') }}

