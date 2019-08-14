# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


def _read_file(file):
    with open(file, 'r') as f:
        return f.read()


def _write_file(filename, content):
    with open(filename, 'w') as f:
        return f.write(content)
