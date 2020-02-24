# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import hashlib
from os.path import exists, join

import jinja2

from pystencils.cpu.cpujit import get_cache_config

_hash = hashlib.md5


def read_template_from_file(file):
    return jinja2.Template(read_file(file), undefined=jinja2.StrictUndefined)


def read_file(file):
    with open(file, 'r') as f:
        content = f.read()
    return content


def write_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)


def write_cached_content(content, suffix):
    filename = join(get_cache_config()['object_cache'], _hash(content.encode()).hexdigest() + suffix)
    if not exists(filename):
        with open(filename, 'w') as f:
            f.write(content)
    return filename
