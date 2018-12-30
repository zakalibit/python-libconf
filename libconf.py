#!/usr/bin/python

from __future__ import absolute_import, division, print_function

import sys
import os
import codecs
import collections
import io
import re

# Define an isstr() and isint() that work on both Python2 and Python3.
# See http://stackoverflow.com/questions/11301138
try:
    basestring  # attempt to evaluate basestring

    def isstr(s):
        return isinstance(s, basestring)

    def isint(i):
        return isinstance(i, (int, long))
except NameError:

    def isstr(s):
        return isinstance(s, str)

    def isint(i):
        return isinstance(i, int)

# Bounds to determine when an "L" suffix should be used during dump().
SMALL_INT_MIN = -2**31
SMALL_INT_MAX = 2**31 - 1

ESCAPE_SEQUENCE_RE = re.compile(r'''
    ( \\x..            # 2-digit hex escapes
    | \\[\\'"abfnrtv]  # Single-character escapes
    )''', re.UNICODE | re.VERBOSE)

SKIP_RE = re.compile(r'\s+|#.*$|//.*$|/\*(.|\n)*?\*/', re.MULTILINE)
UNPRINTABLE_CHARACTER_RE = re.compile(r'[\x00-\x1F\x7F]')


# ${var.name} substitution with in the file or env
RE_VAR = re.compile(r'\${(?P<mvar>[\w\._-]+)}')

## dictionary value lookup, using key, or a path through the object graph
def jpath_find(path, dictionary):
    path_parts = path.split('.')
    key = path_parts[-1]
    key_path = path_parts[:-1]

    ## grab a value on the topmost level if exists
    value = ((key in dictionary) and dictionary[key]) or None

    ## scan dict tree for best match
    current_node = dictionary
    for p in key_path:
        if p not in current_node: break
        ## check if key value exists at current level
        if key in current_node[p]:
            value = current_node[p][key]

        if isinstance(current_node[p], dict):
           current_node = current_node[p]
        else: break

    return value

## lookups a key in a dictionary, key can be a valid json path
def resolve_config_value(path, dictionary):
    value = jpath_find(path, dictionary)
    ## if not found in the config at any level, try env
    if value is None and (path in os.environ):
        value = os.environ[path]
    if value is None:
        raise Exception('{} can not be resolved...'.format(path))
    return value


def parse_substitutions(value, cfg):
    org_value = value
    ## find all ${var}
    res = RE_VAR.findall(value)
    ## handle nested substitutions
    max_nested_subs = 7
    while res and max_nested_subs:
        max_nested_subs -= 1
        for m in res:
            resolved_value = resolve_config_value(m, cfg)
            pattern = '${'+m+'}'
            if value == pattern:
                ## value is a full match
                value = resolved_value
            else:
                ## value contains a macth
                value = value.replace(pattern, str(resolved_value))

        res = isinstance(value, str) and RE_VAR.findall(value)

    if res:
        raise Exception('Max nested substitution limit exceded for: {}'.format(org_value))
    return value


if sys.version_info[0] < 3:
    PY_STR_ENCODING = unicode
else:
    PY_STR_ENCODING = bytes


def deunicodify_hook(pairs, cfg):
    global PY_STR_ENCODING

    new_pairs = []
    for key, value in pairs:
        if isinstance(value, PY_STR_ENCODING):
            value = value.encode('utf-8')
        if isinstance(key, PY_STR_ENCODING):
            key = key.encode('utf-8')

        if isinstance(value, str):
            value = parse_substitutions(value, cfg)
        elif isinstance(value, list):
            nv = []
            for l in value:
                if isinstance(l, PY_STR_ENCODING):
                    l = l.encode('utf-8')
                if isinstance(l, str):
                    l = parse_substitutions(l, cfg)
                nv.append(l)
            value = nv

        new_pairs.append((key, value))
    return collections.OrderedDict(new_pairs)

# load() logic
##############

def decode_escapes(s):
    '''Unescape libconfig string literals'''
    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)


class AttrDict(collections.OrderedDict):
    '''OrderedDict subclass giving access to string keys via attribute access

    This class derives from collections.OrderedDict. Thus, the original
    order of the config entries in the input stream is maintained.
    '''

    def __getattr__(self, attr):
        if attr == '_OrderedDict__root':
            # Work around Python2's OrderedDict weirdness.
            raise AttributeError("AttrDict has no attribute %r" % attr)
        return self.__getitem__(attr)


class ConfigParseError(RuntimeError):
    '''Exception class raised on errors reading the libconfig input'''
    pass


class ConfigSerializeError(TypeError):
    '''Exception class raised on errors serializing a config object'''
    pass


class Token(object):
    '''Base class for all tokens produced by the libconf tokenizer'''
    def __init__(self, type, text, filename, row, column):
        self.type = type
        self.text = text
        self.filename = filename
        self.row = row
        self.column = column

    def __str__(self):
        return "%r in %r, row %d, column %d" % (
            self.text, self.filename, self.row, self.column)


class FltToken(Token):
    '''Token subclass for floating point values'''
    def __init__(self, *args, **kwargs):
        super(FltToken, self).__init__(*args, **kwargs)
        self.value = float(self.text)


class IntToken(Token):
    '''Token subclass for integral values'''
    def __init__(self, *args, **kwargs):
        super(IntToken, self).__init__(*args, **kwargs)
        self.is_long = self.text.endswith('L')
        self.is_hex = (self.text[1:2].lower() == 'x')
        self.value = int(self.text.rstrip('L'), 0)


class BoolToken(Token):
    '''Token subclass for booleans'''
    def __init__(self, *args, **kwargs):
        super(BoolToken, self).__init__(*args, **kwargs)
        self.value = (self.text[0].lower() == 't')


class StrToken(Token):
    '''Token subclass for strings'''
    def __init__(self, *args, **kwargs):
        super(StrToken, self).__init__(*args, **kwargs)
        self.value = decode_escapes(self.text[1:-1])


def compile_regexes(token_map):
    return [(cls, type, re.compile(regex))
            for cls, type, regex in token_map]


class Tokenizer:
    '''Tokenize an input string

    Typical usage:

        tokens = list(Tokenizer("<memory>").tokenize("""a = 7; b = ();"""))

    The filename argument to the constructor is used only in error messages, no
    data is loaded from the file. The input data is received as argument to the
    tokenize function, which yields tokens or throws a ConfigParseError on
    invalid input.

    Include directives are not supported, they must be handled at a higher
    level (cf. the TokenStream class).
    '''

    token_map = compile_regexes([
        (FltToken,  'float',     r'([-+]?(\d+)?\.\d*([eE][-+]?\d+)?)|'
                                 r'([-+]?(\d+)(\.\d*)?[eE][-+]?\d+)'),
        (IntToken,  'hex64',     r'0[Xx][0-9A-Fa-f]+(L(L)?)'),
        (IntToken,  'hex',       r'0[Xx][0-9A-Fa-f]+'),
        (IntToken,  'integer64', r'[-+]?[0-9]+L(L)?'),
        (IntToken,  'integer',   r'[-+]?[0-9]+'),
        (BoolToken, 'boolean',   r'(?i)(true|false)\b'),
        (StrToken,  'string',    r'"([^"\\]|\\.)*"'),
        (Token,     'name',      r'[A-Za-z\*][-A-Za-z0-9_\*]*'),
        (Token,     '}',         r'\}'),
        (Token,     '{',         r'\{'),
        (Token,     ')',         r'\)'),
        (Token,     '(',         r'\('),
        (Token,     ']',         r'\]'),
        (Token,     '[',         r'\['),
        (Token,     ',',         r','),
        (Token,     ';',         r';'),
        (Token,     '=',         r'='),
        (Token,     ':',         r':'),
    ])

    def __init__(self, filename):
        self.filename = filename
        self.row = 1
        self.column = 1

    def tokenize(self, string):
        '''Yield tokens from the input string or throw ConfigParseError'''
        pos = 0
        while pos < len(string):
            m = SKIP_RE.match(string, pos=pos)
            if m:
                skip_lines = m.group(0).split('\n')
                if len(skip_lines) > 1:
                    self.row += len(skip_lines) - 1
                    self.column = 1 + len(skip_lines[-1])
                else:
                    self.column += len(skip_lines[0])

                pos = m.end()
                continue

            for cls, type, regex in self.token_map:
                m = regex.match(string, pos=pos)
                if m:
                    yield cls(type, m.group(0),
                              self.filename, self.row, self.column)
                    self.column += len(m.group(0))
                    pos = m.end()
                    break
            else:
                raise ConfigParseError(
                    "Couldn't load config in %r row %d, column %d: %r" %
                    (self.filename, self.row, self.column,
                     string[pos:pos+20]))


class TokenStream:
    '''Offer a parsing-oriented view on tokens

    Provide several methods that are useful to parsers, like ``accept()``,
    ``expect()``, ...

    The ``from_file()`` method is the preferred way to read input files, as
    it handles include directives, which the ``Tokenizer`` class does not do.
    '''

    def __init__(self, tokens):
        self.position = 0
        self.tokens = list(tokens)

    @classmethod
    def from_file(cls, f, filename=None, includedir='', seenfiles=None):
        '''Create a token stream by reading an input file

        Read tokens from `f`. If an include directive ('@include "file.cfg"')
        is found, read its contents as well.

        The `filename` argument is used for error messages and to detect
        circular imports. ``includedir`` sets the lookup directory for included
        files.  ``seenfiles`` is used internally to detect circular includes,
        and should normally not be supplied by users of is function.
        '''

        if filename is None:
            filename = getattr(f, 'name', '<unknown>')
        if seenfiles is None:
            seenfiles = set()

        if filename in seenfiles:
            raise ConfigParseError("Circular include: %r" % (filename,))
        seenfiles = seenfiles | {filename}  # Copy seenfiles, don't alter it.

        tokenizer = Tokenizer(filename=filename)
        lines = []
        tokens = []
        for line in f:
            m = re.match(r'@include "(.*)"$', line.strip())
            if m:
                tokens.extend(tokenizer.tokenize(''.join(lines)))
                lines = [re.sub(r'\S', ' ', line)]

                includefilename = decode_escapes(m.group(1))
                includefilename = os.path.join(includedir, includefilename)
                try:
                    includefile = open(includefilename, "r")
                except IOError:
                    raise ConfigParseError("Could not open include file %r" %
                                           (includefilename,))

                with includefile:
                    includestream = cls.from_file(includefile,
                                                  filename=includefilename,
                                                  includedir=includedir,
                                                  seenfiles=seenfiles)
                tokens.extend(includestream.tokens)

            else:
                lines.append(line)

        tokens.extend(tokenizer.tokenize(''.join(lines)))
        return cls(tokens)


    def rewind(self):
        self.position = 0
        return self


    def peek(self):
        '''Return (but do not consume) the next token

        At the end of input, ``None`` is returned.
        '''

        if self.position >= len(self.tokens):
            return None

        return self.tokens[self.position]

    def accept(self, *args):
        '''Consume and return the next token if it has the correct type

        Multiple token types (as strings, e.g. 'integer64') can be given
        as arguments. If the next token is one of them, consume and return it.

        If the token type doesn't match, return None.
        '''

        token = self.peek()
        if token is None:
            return None

        for arg in args:
            if token.type == arg:
                self.position += 1
                return token

        return None

    def expect(self, *args):
        '''Consume and return the next token if it has the correct type

        Multiple token types (as strings, e.g. 'integer64') can be given
        as arguments. If the next token is one of them, consume and return it.

        If the token type doesn't match, raise a ConfigParseError.
        '''

        t = self.accept(*args)
        if t is not None:
            return t

        self.error("expected: %r" % (args,))

    def error(self, msg):
        '''Raise a ConfigParseError at the current input position'''
        if self.finished():
            raise ConfigParseError("Unexpected end of input; %s" % (msg,))
        else:
            t = self.peek()
            raise ConfigParseError("Unexpected token %s; %s" % (t, msg))

    def finished(self):
        '''Return ``True`` if the end of the token stream is reached.'''
        return self.position >= len(self.tokens)


class Parser:
    '''Recursive descent parser for libconfig files

    Takes a ``TokenStream`` as input, the ``parse()`` method then returns
    the config file data in a ``json``-module-style format.
    '''

    def __init__(self, tokenstream, object_pairs_hook=None):
        self.tokens = tokenstream
        self.object_pairs_hook = object_pairs_hook

    def parse(self):
        result = self.setting_list_or_empty()
        if not self.tokens.finished():
            raise ConfigParseError("Expected end of input but found %s" %
                                   (self.tokens.peek(),))

        return result


    def setting_list_or_empty(self):
        result = AttrDict()
        while True:
            s = self.setting()
            if s is None:
                return result

            if self.object_pairs_hook is None:
                result[s[0]] = s[1]
            else:
                result.update(self.object_pairs_hook([s]))


    def setting(self):
        name = self.tokens.accept('name')
        if name is None:
            return None

        self.tokens.expect(':', '=')

        value = self.value()
        if value is None:
            self.tokens.error("expected a value")

        self.tokens.accept(';', ',')

        return (name.text, value)

    def value(self):
        acceptable = [self.scalar_value, self.array, self.list, self.group]
        return self._parse_any_of(acceptable)

    def scalar_value(self):
        # This list is ordered so that more common tokens are checked first.
        acceptable = [self.string, self.boolean, self.integer, self.float,
                      self.hex, self.integer64, self.hex64]
        return self._parse_any_of(acceptable)

    def value_list_or_empty(self):
        return tuple(self._comma_separated_list_or_empty(self.value))

    def scalar_value_list_or_empty(self):
        return self._comma_separated_list_or_empty(self.scalar_value)

    def array(self):
        return self._enclosed_block('[', self.scalar_value_list_or_empty, ']')

    def list(self):
        return self._enclosed_block('(', self.value_list_or_empty, ')')

    def group(self):
        return self._enclosed_block('{', self.setting_list_or_empty, '}')

    def boolean(self):
        return self._create_value_node('boolean')

    def integer(self):
        return self._create_value_node('integer')

    def integer64(self):
        return self._create_value_node('integer64')

    def hex(self):
        return self._create_value_node('hex')

    def hex64(self):
        return self._create_value_node('hex64')

    def float(self):
        return self._create_value_node('float')

    def string(self):
        t_first = self.tokens.accept('string')
        if t_first is None:
            return None

        values = [t_first.value]
        while True:
            t = self.tokens.accept('string')
            if t is None:
                break
            values.append(t.value)

        return ''.join(values)

    def _create_value_node(self, tokentype):
        t = self.tokens.accept(tokentype)
        if t is None:
            return None

        return t.value

    def _parse_any_of(self, nonterminals):
        for fun in nonterminals:
            result = fun()
            if result is not None:
                return result

        return None

    def _comma_separated_list_or_empty(self, nonterminal):
        values = []
        first = True
        while True:
            v = nonterminal()
            if v is None:
                if first:
                    return []
                else:
                    self.tokens.error("expected value after ','")

            values.append(v)
            if not self.tokens.accept(','):
                return values

            first = False

    def _enclosed_block(self, start, nonterminal, end):
        if not self.tokens.accept(start):
            return None
        result = nonterminal()
        self.tokens.expect(end)
        return result


def load(f, filename=None, includedir=''):
    '''Load the contents of ``f`` (a file-like object) to a Python object

    The returned object is a subclass of ``dict`` that exposes string keys as
    attributes as well.

    Example:

        >>> with open('test/example.cfg') as f:
        ...     config = libconf.load(f)
        >>> config['window']['title']
        'libconfig example'
        >>> config.window.title
        'libconfig example'
    '''

    if isinstance(f.read(0), bytes):
        raise TypeError("libconf.load() input file must by unicode")

    tokenstream = TokenStream.from_file(f,
                                        filename=filename,
                                        includedir=includedir)
    cfg = Parser(tokenstream).parse()
    return Parser(tokenstream.rewind(), object_pairs_hook=lambda p: deunicodify_hook(p, cfg)).parse()


def loads(string, filename=None, includedir=''):
    '''Load the contents of ``string`` to a Python object

    The returned object is a subclass of ``dict`` that exposes string keys as
    attributes as well.

    Example:

        >>> config = libconf.loads('window: { title: "libconfig example"; };')
        >>> config['window']['title']
        'libconfig example'
        >>> config.window.title
        'libconfig example'
    '''

    try:
        f = io.StringIO(string)
    except TypeError:
        raise TypeError("libconf.loads() input string must by unicode")

    return load(f, filename=filename, includedir=includedir)


# dump() logic
##############

def dump_int(i):
    '''Stringize ``i``, append 'L' if ``i`` is exceeds the 32-bit int range'''
    return str(i) + ('' if SMALL_INT_MIN <= i <= SMALL_INT_MAX else 'L')


def dump_string(s):
    '''Stringize ``s``, adding double quotes and escaping as necessary

    Backslash escape backslashes, double quotes, ``\f``, ``\n``, ``\r``, and
    ``\t``. Escape all remaining unprintable characters in ``\xFF``-style.
    The returned string will be surrounded by double quotes.
    '''

    s = (s.replace('\\', '\\\\')
          .replace('"', '\\"')
          .replace('\f', r'\f')
          .replace('\n', r'\n')
          .replace('\r', r'\r')
          .replace('\t', r'\t'))
    s = UNPRINTABLE_CHARACTER_RE.sub(
            lambda m: r'\x{:02x}'.format(ord(m.group(0))),
            s)
    return '"' + s + '"'


def dump_value(key, value, f, indent=0):
    '''Save a value of any libconfig type

    This function serializes takes ``key`` and ``value`` and serializes them
    into ``f``. If ``key`` is ``None``, a list-style output is produced.
    Otherwise, output has ``key = value`` format.
    '''

    spaces = ' ' * indent

    if key is None:
        key_prefix = ''
        key_prefix_nl = ''
    else:
        key_prefix = key + ' = '
        key_prefix_nl = key + ' =\n' + spaces

    if isinstance(value, dict):
        f.write(u'{}{}{{\n'.format(spaces, key_prefix_nl))
        dump_dict(value, f, indent + 4)
        f.write(u'{}}}'.format(spaces))
    elif isinstance(value, tuple):
        f.write(u'{}{}(\n'.format(spaces, key_prefix_nl))
        dump_collection(value, f, indent + 4)
        f.write(u'\n{})'.format(spaces))
    elif isinstance(value, list):
        f.write(u'{}{}[\n'.format(spaces, key_prefix_nl))
        dump_collection(value, f, indent + 4)
        f.write(u'\n{}]'.format(spaces))
    elif isstr(value):
        f.write(u'{}{}{}'.format(spaces, key_prefix, dump_string(value)))
    elif isint(value):
        f.write(u'{}{}{}'.format(spaces, key_prefix, dump_int(value)))
    elif isinstance(value, float):
        f.write(u'{}{}{}'.format(spaces, key_prefix, value))
    else:
        raise ConfigSerializeError("Can not serialize object %r of type %s" %
                                   (value, type(value)))


def dump_collection(cfg, f, indent=0):
    '''Save a collection of attributes'''

    for i, value in enumerate(cfg):
        dump_value(None, value, f, indent)
        if i < len(cfg) - 1:
            f.write(u',\n')


def dump_dict(cfg, f, indent=0):
    '''Save a dictionary of attributes'''

    for key in cfg:
        if not isstr(key):
            raise ConfigSerializeError("Dict keys must be strings: %r" %
                                       (key,))
        dump_value(key, cfg[key], f, indent)
        f.write(u';\n')


def dumps(cfg):
    '''Serialize ``cfg`` into a libconfig-formatted ``str``

    ``cfg`` must be a ``dict`` with ``str`` keys and libconf-supported values
    (numbers, strings, booleans, possibly nested dicts, lists, and tuples).

    Returns the formatted string.
    '''

    str_file = io.StringIO()
    dump(cfg, str_file)
    return str_file.getvalue()


def dump(cfg, f):
    '''Serialize ``cfg`` as a libconfig-formatted stream into ``f``

    ``cfg`` must be a ``dict`` with ``str`` keys and libconf-supported values
    (numbers, strings, booleans, possibly nested dicts, lists, and tuples).

    ``f`` must be a ``file``-like object with a ``write()`` method.
    '''

    if not isinstance(cfg, dict):
        raise ConfigSerializeError(
                'dump() requires a dict as input, not %r of type %r' %
                (cfg, type(cfg)))

    dump_dict(cfg, f, 0)


# main(): small example of how to use libconf
#############################################

def main():
    '''Open the libconfig file specified by sys.argv[1] and pretty-print it'''
    global output
    if len(sys.argv[1:]) == 1:
        with io.open(sys.argv[1], 'r', encoding='utf-8') as f:
            output = load(f)
    else:
        output = load(sys.stdin)

    dump(output, sys.stdout)


if __name__ == '__main__':
    main()
