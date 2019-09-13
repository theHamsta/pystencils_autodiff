# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import jinja2


def read_template_from_file(file):
    return jinja2.Template(read_file(file))


def read_file(file):
    with open(file, 'r') as f:
        content = f.read()
    return content


def write_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)
