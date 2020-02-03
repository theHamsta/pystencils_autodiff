/*
 * walberla_user_defintions.tmpl.hpp
 * Copyright (C) 2020 Stephan Seitz <stephan.seitz@fau.de>
 *
 * Distributed under terms of the GPLv3 license.
 */

#pragma once


{% for header in headers -%}
#include {{ header }}
{% endfor %}

namespace walberla_user {
using namespace walberla;
{% if with_lbm %}
using LatticeModel_T = lbm::{{ lb_model_name }};
using Stencil_T = LatticeModel_T::Stencil;
using CommunicationStencil_T = LatticeModel_T::CommunicationStencil;
{% endif %}
using flag_t = {{ flag_field_type }};
using FlagField_T = FlagField<flag_t>;

} // namespace walberla_user
