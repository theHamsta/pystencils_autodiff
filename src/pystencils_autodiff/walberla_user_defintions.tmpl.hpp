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
using LatticeModel_T = lbm::{{ lb_model_name }};
using Stencil_T = LatticeModel_T::Stencil;
using CommunicationStencil_T = LatticeModel_T::CommunicationStencil;
using PdfField_T = lbm::PdfField<LatticeModel_T>;
using flag_t = {{ flag_field_type }};
using FlagField_T = FlagField<flag_t>;

using PdfField_T = lbm::PdfField<LatticeModel_T>;
using VectorField_T = GhostLayerField<real_t, LatticeModel_T::Stencil::D>;
using ScalarField_T = GhostLayerField<real_t, 1>;
} // namespace walberla_user
