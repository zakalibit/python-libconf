# -*- coding: utf-8 -*-

import os
import io
import sys
import libconf


CURDIR = os.path.abspath(os.path.dirname(__file__))


def test_substitutions():
    example_file = os.path.join(CURDIR, 'example_subs.cfg')
    with io.open(example_file, 'r', encoding='utf-8') as f:
        c = libconf.load(f, includedir=CURDIR)
    
        assert c['capabilities']['title'] == c['window']['title'] 
        assert c['capabilities']['version'] == c['version']

        print libconf.dumps(c)
