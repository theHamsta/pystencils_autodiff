# -*- coding: utf-8 -*-
#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import keyword
import string

from unidecode import unidecode

CXX_IDENTIFIERS = ["alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept",
                   "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t",
                   "char32_t", "class", "compl", "concept", "const", "consteval", "constexpr", "constinit",
                   "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype", "default", "delete", "do",
                   "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float", "for",
                   "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not",
                   "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr",
                   "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
                   "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this",
                   "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned",
                   "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "override", "final",
                   "import", "module", "transaction_safe", "transaction_safe_dynamic"]


def escape_python_indentifiers(ident, fill='_', is_keyword_predicate=keyword.iskeyword, ascii_only=False):
    """
    Escapes the column header string to make it a valid Python/C identifier.

    Adapted from https://github.com/FrankSalad/ipython-csvmagic/blob/master/csvtools.py (MIT)
    """
    # Reference: https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    # Starts with a number
    if ident[0] in string.digits:
        ident = fill + ident

    if ascii_only:
        ident = unidecode(ident)

    if is_keyword_predicate(ident):
        ident = fill + ident

    def replace_char(c):
        if not c.isalnum() or (ascii_only and not c.isascii()):
            return fill
        else:
            return c

    return ''.join(map(replace_char, ident))


def escape_cxx_indentifiers(indent, fill='_', ascii_only=True):
    return escape_python_indentifiers(indent, fill, lambda x: x in CXX_IDENTIFIERS, ascii_only=ascii_only)
