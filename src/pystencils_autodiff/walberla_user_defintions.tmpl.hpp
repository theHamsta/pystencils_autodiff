/*
 * walberla_user_defintions.tmpl.hpp
 * Copyright (C) 2020 Stephan Seitz <stephan.seitz@fau.de>
 *
 * Distributed under terms of the GPLv3 license.
 */

#pragma once

#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/gui/Connection.h"
#include "lbm/vtk/VTKOutput.h"

namespace walberla_user {
using namespace walberla;
using LatticeModel_T = lbm::LbCodeGenerationExample_LatticeModel;
using Stencil_T = LatticeModel_T::Stencil;
using CommunicationStencil_T = LatticeModel_T::CommunicationStencil_T;
using lPdfField_T = bm::PdfField<LatticeModel_T>;
using flag_t = walberla::uint8_t;
using FlagField_T FlagField<flag_t>;

using PdfField_T = lbm::PdfField<LatticeModel_T>;
typedef VectorField_T = GhostLayerField<real_t, LatticeModel_T::Stencil::D>;
typedef ScalarField_T GhostLayerField<real_t, 1>;
} // namespace walberla_user
