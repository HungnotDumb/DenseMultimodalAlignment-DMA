#   Copyright 2014 Darsh Ranjan
#
#   This file is part of python-plyfile.
#
#   python-plyfile is free software: you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   python-plyfile is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with python-plyfile.  If not, see
#       <http://www.gnu.org/licenses/>.

from itertools import islice as _islice

import numpy as _np
from sys import byteorder as _byteorder


try:
    _range = xrange
except NameError:
    _range = range


# Many-many relation
_data_type_relation = [
    ('int8', 'i1'),
    ('char', 'i1'),
    ('uint8', 'u1'),
    ('uchar', 'b1'),
    ('uchar', 'u1'),
    ('int16', 'i2'),
    ('short', 'i2'),
    ('uint16', 'u2'),
    ('ushort', 'u2'),
    ('int32', 'i4'),
    ('int', 'i4'),
    ('uint32', 'u4'),
    ('uint', 'u4'),
    ('float32', 'f4'),
    ('float', 'f4'),
    ('float64', 'f8'),
    ('double', 'f8')
]

_data_types = dict(_data_type_relation)
_data_type_reverse = dict((b, a) for (a, b) in _data_type_relation)

_types_list = []
_types_set = set()
for (_a, _b) in _data_type_relation:
    if _a not in _types_set:
        _types_list.append(_a)
        _types_set.add(_a)
    if _b not in _types_set:
        _types_list.append(_b)
        _types_set.add(_b)


_byte_order_map = {
    'ascii': '=',
    'binary_little_endian': '<',
    'binary_big_endian': '>'
}

_byte_order_reverse = {
    '<': 'binary_little_endian',
    '>': 'binary_big_endian'
}

_native_byte_order = {'little': '<', 'big': '>'}[_byteorder]


def _lookup_type(type_str):
    if type_str not in _data_type_reverse:
        try:
            type_str = _data_types[type_str]
        except KeyError:
            raise ValueError("field type %r not in %r" %
                             (type_str, _types_list))

    return _data_type_reverse[type_str]


def _split_line(line, n):
    fields = line.split(None, n)
    if len(fields) == n:
        fields.append('')

    assert len(fields) == n + 1

    return fields


def make2d(array, cols=None, dtype=None):
    '''
    Make a 2D array from an array of arrays.  The `cols' and `dtype'
    arguments can be omitted if the array is not empty.

    '''
    if (cols is None or dtype is None) and not len(array):
        raise RuntimeError("cols and dtype must be specified for empty "
                           "array")

    if cols is None:
        cols = len(array[0])

    if dtype is None:
        dtype = array[0].dtype

    return _np.fromiter(array, [('_', dtype, (cols,))],
                        count=len(array))['_']


class PlyParseError(Exception):

    '''
    Raised when a PLY file cannot be parsed.

    The attributes `element', `row', `prope