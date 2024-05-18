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

    The attributes `element', `row', `property', and `message' give
    additional information.

    '''

    def __init__(self, message, element=None, row=None, prop=None):
        self.message = message
        self.element = element
        self.row = row
        self.prop = prop

        s = ''
        if self.element:
            s += 'element %r: ' % self.element.name
        if self.row is not None:
            s += 'row %d: ' % self.row
        if self.prop:
            s += 'property %r: ' % self.prop.name
        s += self.message

        Exception.__init__(self, s)

    def __repr__(self):
        return ('PlyParseError(%r, element=%r, row=%r, prop=%r)' %
                self.message, self.element, self.row, self.prop)


class PlyData(object):

    '''
    PLY file header and data.

    A PlyData instance is created in one of two ways: by the static
    method PlyData.read (to read a PLY file), or directly from __init__
    given a sequence of elements (which can then be written to a PLY
    file).

    '''

    def __init__(self, elements=[], text=False, byte_order='=',
                 comments=[], obj_info=[]):
        '''
        elements: sequence of PlyElement instances.

        text: whether the resulting PLY file will be text (True) or
            binary (False).

        byte_order: '<' for little-endian, '>' for big-endian, or '='
            for native.  This is only relevant if `text' is False.

        comments: sequence of strings that will be placed in the header
            between the 'ply' and 'format ...' lines.

        obj_info: like comments, but will be placed in the header with
            "obj_info ..." instead of "comment ...".

        '''
        if byte_order == '=' and not text:
            byte_order = _native_byte_order

        self.byte_order = byte_order
        self.text = text

        self.comments = list(comments)
        self.obj_info = list(obj_info)
        self.elements = elements

    def _get_elements(self):
        return self._elements

    def _set_elements(self, elements):
        self._elements = tuple(elements)
        self._index()

    elements = property(_get_elements, _set_elements)

    def _get_byte_order(self):
        return self._byte_order

    def _set_byte_order(self, byte_order):
        if byte_order not in ['<', '>', '=']:
            raise ValueError("byte order must be '<', '>', or '='")

        self._byte_order = byte_order

    byte_order = property(_get_byte_order, _set_byte_order)

    def _index(self):
        self._element_lookup = dict((elt.name, elt) for elt in
                                    self._elements)
        if len(self._element_lookup) != len(self._elements):
            raise ValueError("two elements with same name")

    @staticmethod
    def _parse_header(stream):
        '''
        Parse a PLY header from a readable file-like stream.

        '''
        lines = []
        comments = {'comment': [], 'obj_info': []}
        while True:
            line = stream.readline().decode('ascii').strip()
            fields = _split_line(line, 1)

            if fields[0] == 'end_header':
                break

            elif fields[0] in comments.keys():
                lines.append(fields)
            else:
                lines.append(line.split())

        a = 0
        if lines[a] != ['ply']:
            raise PlyParseError("expected 'ply'")

        a += 1
        while lines[a][0] in comments.keys():
            comments[lines[a][0]].append(lines[a][1])
            a += 1

        if lines[a][0] != 'format':
            raise PlyParseError("expected 'format'")

        if lines[a][2] != '1.0':
            raise PlyParseError("expected version '1.0'")

        if len(lines[a]) != 3:
            raise PlyParseError("too many fields after 'format'")

        fmt = lines[a][1]

        if fmt not in _byte_order_map:
            raise PlyParseError("don't understand format %r" % fmt)

        byte_order = _byte_order_map[fmt]
        text = fmt == 'ascii'

        a += 1
        while a < len(lines) and lines[a][0] in comments.keys():
            comments[lines[a][0]].append(lines[a][1])
            a += 1

        return PlyData(PlyElement._parse_multi(lines[a:]),
                       text, byte_order,
                       comments['comment'], comments['obj_info'])

    @staticmethod
    def read(stream):
        '''
        Read PLY data from a readable file-like object or filename.

        '''
        (must_close, stream) = _open_stream(stream, 'read')
        try:
            data = PlyData._parse_header(stream)
            for elt in data:
                elt._read(stream, data.text, data.byte_order)
        finally:
            if must_close:
                stream.close()

        return data

    def write(self, stream):
        '''
        Write PLY data to a writeable file-like object or filename.

        '''
        (must_close, stream) = _open_stream(stream, 'write')
        try:
            stream.write(self.header.encode('ascii'))
            stream.write(b'\r\n')
            for elt in self:
                elt._write(stream, self.text, self.byte_order)
        finally:
            if must_close:
                stream.close()

    @property
    def header(self):
        '''
        Provide PLY-formatted metadata for the instance.

        '''
        lines = ['ply']

        if self.text:
            lines.append('format ascii 1.0')
        else:
            lines.append('format ' +
                         _byte_order_reverse[self.byte_order] +
                         ' 1.0')

        # Some information is lost here, since all comments are placed
        # between the 'format' line and the first element.
        for c in self.comments:
            lines.append('comment ' + c)

        for c in self.obj_info:
            lines.append('obj_info ' + c)

        lines.extend(elt.header for elt in self.elements)
        lines.append('end_header')
        return '\r\n'.join(lines)

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __contains__(self, name):
        return name in self._element_lookup

    def __getitem__(self, name):
        return self._element_lookup[name]

    def __str__(self):
        return self.header

    def __repr__(self):
        return ('PlyData(%r, text=%r, byte_order=%r, '
                'comments=%r, obj_info=%r)' %
                (self.elements, self.text, self.byte_order,
                 self.comments, self.obj_info))


def _open_stream(stream, read_or_write):
    if hasattr(stream, read_or_write):
        return (False, stream)
    try:
        return (True, open(stream, read_or_write[0] + 'b'))
    except TypeError:
        raise RuntimeError("expected open file or filename")


class PlyElement(object):

    '''
    PLY file element.

    A client of this library doesn't normally need to instantiate this
    directly, so the following is only for the sake of documenting the
    internals.

    Creating a PlyElement instance is generally done in one of two ways:
    as a byproduct of PlyData.read (when reading a PLY file) and by
    PlyElement.describe (before writing a PLY file).

    '''

    def __init__(self, name, properties, count, comments=[]):
        '''
        This is not part of the public interface.  The preferred methods
        of obtaining PlyElement instances are PlyData.read (to read from
        a file) and PlyElement.describe (to construct from a numpy
        array).

        '''
        self._name = str(name)
        self._check_name()
        self._count = count

        self._properties = tuple(properties)
        self._index()

        self.comments = list(comments)

        self._have_list = any(isinstance(p, PlyListProperty)
                              for p in self.properties)

    @property
    def count(self):
        return self._count

    def _get_data(self):
        return self._data

    def _set_data(self, data):
        self._data 