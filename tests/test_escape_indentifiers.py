# -*- coding: utf-8 -*-
#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from pystencils_autodiff.escape_identifiers import escape_cxx_indentifiers


def test_escape_indentifiers():
    assert escape_cxx_indentifiers('2áα') == '_2aa'
    assert escape_cxx_indentifiers('foo:0') == 'foo_0'
    assert escape_cxx_indentifiers('##j') == '__j'
