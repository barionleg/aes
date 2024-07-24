"""File formats.

This module implements the read(), iread() and write() functions in ase.io.
For each file format there is an IOFormat object.

There is a dict, ioformats, which stores the objects.

Example
=======

The xyz format is implemented in the ase/io/xyz.py file which has a
read_xyz() generator and a write_xyz() function.  This and other
information can be obtained from ioformats['xyz'].
"""

import functools
import inspect
import io
import numbers
import os
import re
import sys
import warnings
from importlib import import_module
from pathlib import Path, PurePath
from typing import (IO, Any, Iterable, List, Optional, Sequence, Tuple,
                    Union)
from ase.utils import lazyproperty
from ase.register.pluggables import BasePluggable, Pluggables
from ase.register.listing import ListingView

from ase.atoms import Atoms
from ase.parallel import parallel_function, parallel_generator
from ase.utils import string2index
import numpy as np

PEEK_BYTES = 50000


class UnknownFileTypeError(Exception):
    pass


def normalize_pbc(pbc: Union[str, bytes, np.ndarray, List, Tuple]) \
        -> np.ndarray:
    """
    >>> normalize_pbc("101")
    array([ True, False,  True])
    >>> normalize_pbc(b"110")
    array([ True, True,  False])
    >>> normalize_pbc("110")
    array([ True, True,  False])
    >>> normalize_pbc([0,0,1])
    array([ False, False,  True])
    """
    if isinstance(pbc, (np.ndarray, List, tuple)):
        pbc = np.asarray(pbc, dtype=bool)
    else:
        if isinstance(pbc, str):
            pbc = str.encode("ascii")
        if max(pbc) > 1:
            pbc = np.fromiter((i - ord('0') for i in pbc),
                              dtype=bool, count=3)
        else:
            pbc = np.fromiter((i for i in pbc), dtype=bool, count=3)
    assert len(pbc) == 3
    return pbc


class IOFormat(BasePluggable):

    class_type = 'io_formats'

    def __init__(self, name: str, desc: str, code: str, module_name: str,
                 encoding: str = None,
                 allowed_pbc: Optional[
                     List[Union[str, bytes, np.ndarray, List, Tuple]]
                 ] = None) -> None:
        self.name = name
        self.description = desc
        assert len(code) == 2
        assert code[0] in list('+1')
        assert code[1] in list('BFS')
        self.code = code
        if allowed_pbc:
            allowed_pbc = [normalize_pbc(i) for i in allowed_pbc]
        self.allowed_pbc = allowed_pbc
        self.module_name = module_name
        self.encoding = encoding

        # (To be set by define_io_format())
        self.extensions: List[str] = []
        self.globs: List[str] = []
        self.magic: List[str] = []
        self.magic_regex: Optional[bytes] = None

    def __getstate__(self):
        """ Just avoid de/serializing the plugin, save its name instead """
        out = self.__dict__.copy()
        if 'plugin' in out:
            out['plugin'] = out['plugin'].name
        return out

    def __setstate__(self, state):
        """ Just avoid de/serializing the plugin, save its name instead """
        self.__dict__.update(state)
        if 'plugin' in state:
            self.plugin = ase_plugins.plugins[self.plugin]

    def open(self, fname, mode: str = 'r') -> IO:
        # We might want append mode, too
        # We can allow more flags as needed (buffering etc.)
        if mode not in list('rwa'):
            raise ValueError("Only modes allowed are 'r', 'w', and 'a'")
        if mode == 'r' and not self.can_read:
            raise NotImplementedError('No reader implemented for {} format'
                                      .format(self.name))
        if mode == 'w' and not self.can_write:
            raise NotImplementedError('No writer implemented for {} format'
                                      .format(self.name))
        if mode == 'a' and not self.can_append:
            raise NotImplementedError('Appending not supported by {} format'
                                      .format(self.name))

        if self.isbinary:
            mode += 'b'

        path = Path(fname)
        return path.open(mode, encoding=self.encoding)

    def _buf_as_filelike(self, data: Union[str, bytes]) -> IO:
        encoding = self.encoding
        if encoding is None:
            encoding = 'utf-8'  # Best hacky guess.

        if self.isbinary:
            if isinstance(data, str):
                data = data.encode(encoding)
        else:
            if isinstance(data, bytes):
                data = data.decode(encoding)

        return self._ioclass(data)

    @property
    def _ioclass(self):
        if self.isbinary:
            return io.BytesIO
        else:
            return io.StringIO

    def parse_images(self, data: Union[str, bytes],
                     **kwargs) -> Sequence[Atoms]:
        with self._buf_as_filelike(data) as fd:
            outputs = self.read(fd, **kwargs)
            if self.single:
                assert isinstance(outputs, Atoms)
                return [outputs]
            else:
                return list(self.read(fd, **kwargs))

    def parse_atoms(self, data: Union[str, bytes], **kwargs) -> Atoms:
        images = self.parse_images(data, **kwargs)
        return images[-1]

    @property
    def can_read(self) -> bool:
        return self._readfunc() is not None

    @property
    def can_write(self) -> bool:
        return self._writefunc() is not None

    @property
    def can_append(self) -> bool:
        writefunc = self._writefunc()
        return self.can_write and 'append' in writefunc.__code__.co_varnames

    def __repr__(self) -> str:
        tokens = [f'{name}={value!r}'
                  for name, value in vars(self).items()]
        return 'IOFormat({})'.format(', '.join(tokens))

    def __getitem__(self, i):
        # For compatibility.
        #
        # Historically, the ioformats were listed as tuples
        # with (description, code).  We look like such a tuple.
        return (self.description, self.code)[i]

    @property
    def single(self) -> bool:
        """Whether this format is for a single Atoms object."""
        return self.code[0] == '1'

    @property
    def _formatname(self) -> str:
        return self.name.replace('-', '_')

    def _readfunc(self):
        return getattr(self.module, 'read_' + self._formatname, None)

    def _writefunc(self):
        return getattr(self.module, 'write_' + self._formatname, None)

    @lazyproperty
    def implementation(self):
        return self._readfunc(), self._writefunc()

    @property
    def read(self):
        if not self.can_read:
            self._warn_none('read')
            return None

        return self._read_wrapper

    def _read_wrapper(self, *args, **kwargs):
        function = self._readfunc()
        if function is None:
            self._warn_none('read')
            return None
        if not inspect.isgeneratorfunction(function):
            function = functools.partial(wrap_read_function, function)
        return function(*args, **kwargs)

    def _warn_none(self, action):
        msg = ('Accessing the IOFormat.{action} property on a format '
               'without {action} support will change behaviour in the '
               'future and return a callable instead of None.  '
               'Use IOFormat.can_{action} to check whether {action} '
               'is supported.')
        warnings.warn(msg.format(action=action), FutureWarning)

    def are_such_pbc_allowed(self, pbc: np.ndarray) -> bool:
        if self.allowed_pbc is None:
            return True
        for i in self.allowed_pbc:
            if np.array_equal(i, pbc):
                return True
        return False

    @property
    def write(self):
        if not self.can_write:
            self._warn_none('write')
            return None

        return self._write_wrapper

    def _write_wrapper(self, filename: Union[str, PurePath, IO],
                       images: Union[Atoms, Sequence[Atoms]],
                       *args, **kwargs):
        function = self._writefunc()
        if function is None:
            raise ValueError(f'Cannot write to {self.name}-format')

        if self.allowed_pbc is not None:

            def check_pbc(atoms):
                if not self.are_such_pbc_allowed(atoms.pbc):
                    raise ValueError(f'Cannot write Atoms with PBC {atoms.pbc}'
                                     f'into {self.name}-format.')

            if hasattr(images, "pbc"):
                check_pbc(images)
            else:
                for i in images:
                    check_pbc(i)

        return function(filename, images, *args, **kwargs)

    @property
    def modes(self) -> str:
        modes = ''
        if self.can_read:
            modes += 'r'
        if self.can_write:
            modes += 'w'
        return modes

    def full_description(self) -> str:
        lines = [f'Name:        {self.name}',
                 f'Description: {self.description}',
                 f'Modes:       {self.modes}',
                 f'Encoding:    {self.encoding}',
                 f'Module:      {self.module_name}',
                 f'Code:        {self.code}',
                 f'Extensions:  {self.extensions}',
                 f'Globs:       {self.globs}',
                 f'Magic:       {self.magic}']
        return '\n'.join(lines)

    @property
    def acceptsfd(self) -> bool:
        return self.code[1] != 'S'

    @property
    def isbinary(self) -> bool:
        return self.code[1] == 'B'

    @property
    def module(self):
        try:
            return import_module(self.module_name)
        except ImportError as err:
            raise UnknownFileTypeError(
                f'File format not recognized: {self.name}.  Error: {err}')

    def match_name(self, basename: str) -> bool:
        from fnmatch import fnmatch
        return any(fnmatch(basename, pattern)
                   for pattern in self.globs)

    def match_magic(self, data: bytes) -> bool:
        if self.magic_regex:
            assert not self.magic, 'Define only one of magic and magic_regex'
            match = re.match(self.magic_regex, data, re.M | re.S)
            return match is not None

        from fnmatch import fnmatchcase
        return any(
            fnmatchcase(data, magic + b'*')  # type: ignore[operator, type-var]
            for magic in self.magic
        )

    def info(self, prefix='', opts={}):
        infos = [self.modes, 'single' if self.single else 'multi']
        if self.isbinary:
            infos.append('binary')
        if self.encoding is not None:
            infos.append(self.encoding)
        infostring = '/'.join(infos)

        moreinfo = [infostring]
        if self.extensions:
            moreinfo.append('ext={}'.format('|'.join(self.extensions)))
        if self.globs:
            moreinfo.append('glob={}'.format('|'.join(self.globs)))

        out = f'{prefix}{self.name} [{", ".join(moreinfo)}]: {self.description}'
        if getattr(opts, 'plugin', True):
            out += f'  (from plugin {self.plugin.name})'
        return out


class IOFormatPluggables(Pluggables):

    item_type = IOFormat

    def info(self, prefix='', opts={}, filter=None):
        return f"{prefix}IO Formats:\n" \
               f"{prefix}-----------\n" + \
               super().info(prefix + '  ', opts, filter)

    @lazyproperty
    def by_extension(self):
        return self.view_by('extensions')


format2modulename = {}  # Left for compatibility only.


def define_io_format(name, desc, code, *, module=None, ext=None,
                     glob=None, magic=None, encoding=None,
                     magic_regex=None, external=False,
                     allowed_pbc: Optional[
                         List[Union[str, bytes, np.ndarray, List, Tuple]]
                     ] = None):
    if module is None:
        module = name.replace('-', '_')
        format2modulename[name] = module

    if not external:
        module = 'ase.io.' + module

    def normalize_patterns(strings):
        if strings is None:
            strings = []
        elif isinstance(strings, (str, bytes)):
            strings = [strings]
        else:
            strings = list(strings)
        return strings

    fmt = IOFormat(name, desc, code, module_name=module,
                   encoding=encoding, allowed_pbc=allowed_pbc)
    fmt.extensions = normalize_patterns(ext)
    fmt.globs = normalize_patterns(glob)
    fmt.magic = normalize_patterns(magic)

    if magic_regex is not None:
        fmt.magic_regex = magic_regex

    return fmt


def get_ioformat(name: str) -> IOFormat:
    """Return ioformat object or raise appropriate error."""
    if name not in ioformats:
        raise UnknownFileTypeError(name)
    fmt = ioformats[name]
    # Make sure module is importable, since this could also raise an error.
    fmt.module
    return ioformats[name]


def get_compression(filename: str) -> Tuple[str, Optional[str]]:
    """
    Parse any expected file compression from the extension of a filename.
    Return the filename without the extension, and the extension. Recognises
    ``.gz``, ``.bz2``, ``.xz``.

    >>> get_compression('H2O.pdb.gz')
    ('H2O.pdb', 'gz')
    >>> get_compression('crystal.cif')
    ('crystal.cif', None)

    Parameters
    ==========
    filename: str
        Full filename including extension.

    Returns
    =======
    (root, extension): (str, str or None)
        Filename split into root without extension, and the extension
        indicating compression format. Will not split if compression
        is not recognised.
    """
    # Update if anything is added
    valid_compression = ['gz', 'bz2', 'xz']

    # Use stdlib as it handles most edge cases
    root, compression = os.path.splitext(filename)

    # extension keeps the '.' so remember to remove it
    if compression.strip('.') in valid_compression:
        return root, compression.strip('.')
    else:
        return filename, None


def open_with_compression(filename: str, mode: str = 'r') -> IO:
    """
    Wrapper around builtin `open` that will guess compression of a file
    from the filename and open it for reading or writing as if it were
    a standard file.

    Implemented for ``gz``(gzip), ``bz2``(bzip2) and ``xz``(lzma).

    Supported modes are:
       * 'r', 'rt', 'w', 'wt' for text mode read and write.
       * 'rb, 'wb' for binary read and write.

    Parameters
    ==========
    filename: str
        Path to the file to open, including any extensions that indicate
        the compression used.
    mode: str
        Mode to open the file, same as for builtin ``open``, e.g 'r', 'w'.

    Returns
    =======
    fd: file
        File-like object open with the specified mode.
    """

    # Compressed formats sometimes default to binary, so force text mode.
    if mode == 'r':
        mode = 'rt'
    elif mode == 'w':
        mode = 'wt'
    elif mode == 'a':
        mode = 'at'

    root, compression = get_compression(filename)

    if compression == 'gz':
        import gzip
        return gzip.open(filename, mode=mode)  # type: ignore[return-value]
    elif compression == 'bz2':
        import bz2
        return bz2.open(filename, mode=mode)
    elif compression == 'xz':
        import lzma
        return lzma.open(filename, mode)
    else:
        # Either None or unknown string
        return open(filename, mode)


def is_compressed(fd: io.BufferedIOBase) -> bool:
    """Check if the file object is in a compressed format."""
    compressed = False

    # We'd like to avoid triggering imports unless already imported.
    # Also, Python can be compiled without e.g. lzma so we need to
    # protect against that:
    if 'gzip' in sys.modules:
        import gzip
        compressed = compressed or isinstance(fd, gzip.GzipFile)
    if 'bz2' in sys.modules:
        import bz2
        compressed = compressed or isinstance(fd, bz2.BZ2File)
    if 'lzma' in sys.modules:
        import lzma
        compressed = compressed or isinstance(fd, lzma.LZMAFile)
    return compressed


def wrap_read_function(read, filename, index=None, **kwargs):
    """Convert read-function to generator."""
    if index is None:
        yield read(filename, **kwargs)
    else:
        yield from read(filename, index, **kwargs)


NameOrFile = Union[str, PurePath, IO]


def write(
        filename: NameOrFile,
        images: Union[Atoms, Sequence[Atoms]],
        format: str = None,
        parallel: bool = True,
        append: bool = False,
        **kwargs: Any
) -> None:
    """Write Atoms object(s) to file.

    filename: str or file
        Name of the file to write to or a file descriptor.  The name '-'
        means standard output.
    images: Atoms object or list of Atoms objects
        A single Atoms object or a list of Atoms objects.
    format: str
        Used to specify the file-format.  If not given, the
        file-format will be taken from suffix of the filename.
    parallel: bool
        Default is to write on master only.  Use parallel=False to write
        from all slaves.
    append: bool
        Default is to open files in 'w' or 'wb' mode, overwriting
        existing files.  In some cases opening the file in 'a' or 'ab'
        mode (appending) is useful,
        e.g. writing trajectories or saving multiple Atoms objects in one file.
        WARNING: If the file format does not support multiple entries without
        additional keywords/headers, files created using 'append=True'
        might not be readable by any program! They will nevertheless be
        written without error message.

    The use of additional keywords is format specific. write() may
    return an object after writing certain formats, but this behaviour
    may change in the future.

    """

    if isinstance(filename, PurePath):
        filename = str(filename)

    if isinstance(filename, str):
        fd = None
        if filename == '-':
            fd = sys.stdout
            filename = None  # type: ignore[assignment]
        elif format is None:
            format = filetype(filename, read=False)
            assert isinstance(format, str)
    else:
        fd = filename  # type: ignore[assignment]
        if format is None:
            try:
                format = filetype(filename, read=False)
                assert isinstance(format, str)
            except UnknownFileTypeError:
                format = None
        filename = None  # type: ignore[assignment]

    format = format or 'json'  # default is json

    io = get_ioformat(format)

    return _write(filename, fd, format, io, images,
                  parallel=parallel, append=append, **kwargs)


@parallel_function
def _write(filename, fd, format, io, images, parallel=None, append=False,
           **kwargs):
    if isinstance(images, Atoms):
        images = [images]

    if io.single:
        if len(images) > 1:
            raise ValueError('{}-format can only store 1 Atoms object.'
                             .format(format))
        images = images[0]

    if not io.can_write:
        raise ValueError(f"Can't write to {format}-format")

    # Special case for json-format:
    if format == 'json' and (len(images) > 1 or append):
        if filename is not None:
            return io.write(filename, images, append=append, **kwargs)
        raise ValueError("Can't write more than one image to file-descriptor "
                         'using json-format.')

    if io.acceptsfd:
        open_new = (fd is None)
        try:
            if open_new:
                mode = 'wb' if io.isbinary else 'w'
                if append:
                    mode = mode.replace('w', 'a')
                fd = open_with_compression(filename, mode)
                # XXX remember to re-enable compressed open
                # fd = io.open(filename, mode)
            return io.write(fd, images, **kwargs)
        finally:
            if open_new and fd is not None:
                fd.close()
    else:
        if fd is not None:
            raise ValueError("Can't write {}-format to file-descriptor"
                             .format(format))
        if io.can_append:
            return io.write(filename, images, append=append, **kwargs)
        elif append:
            raise ValueError("Cannot append to {}-format, write-function "
                             "does not support the append keyword."
                             .format(format))
        else:
            return io.write(filename, images, **kwargs)


def read(
        filename: NameOrFile,
        index: Any = None,
        format: Optional[str] = None,
        parallel: bool = True,
        do_not_split_by_at_sign: bool = False,
        **kwargs
) -> Union[Atoms, List[Atoms]]:
    """Read Atoms object(s) from file.

    filename: str or file
        Name of the file to read from or a file descriptor.
    index: int, slice or str
        The last configuration will be returned by default.  Examples:

            * ``index=0``: first configuration
            * ``index=-2``: second to last
            * ``index=':'`` or ``index=slice(None)``: all
            * ``index='-3:'`` or ``index=slice(-3, None)``: three last
            * ``index='::2'`` or ``index=slice(0, None, 2)``: even
            * ``index='1::2'`` or ``index=slice(1, None, 2)``: odd
    format: str
        Used to specify the file-format.  If not given, the
        file-format will be guessed by the *filetype* function.
    parallel: bool
        Default is to read on master and broadcast to slaves.  Use
        parallel=False to read on all slaves.
    do_not_split_by_at_sign: bool
        If False (default) ``filename`` is splitted by at sign ``@``

    Many formats allow on open file-like object to be passed instead
    of ``filename``. In this case the format cannot be auto-detected,
    so the ``format`` argument should be explicitly given."""

    if isinstance(filename, PurePath):
        filename = str(filename)
    if filename == '-':
        filename = sys.stdin
    if isinstance(index, str):
        try:
            index = string2index(index)
        except ValueError:
            pass

    filename, index = parse_filename(filename, index, do_not_split_by_at_sign)
    if index is None:
        index = -1
    format = format or filetype(filename, read=isinstance(filename, str))

    io = get_ioformat(format)
    if isinstance(index, (slice, str)):
        return list(_iread(filename, index, format, io, parallel=parallel,
                           **kwargs))
    else:
        return next(_iread(filename, slice(index, None), format, io,
                           parallel=parallel, **kwargs))


def iread(
        filename: NameOrFile,
        index: Any = None,
        format: str = None,
        parallel: bool = True,
        do_not_split_by_at_sign: bool = False,
        **kwargs
) -> Iterable[Atoms]:
    """Iterator for reading Atoms objects from file.

    Works as the `read` function, but yields one Atoms object at a time
    instead of all at once."""

    if isinstance(filename, PurePath):
        filename = str(filename)

    if isinstance(index, str):
        index = string2index(index)

    filename, index = parse_filename(filename, index, do_not_split_by_at_sign)

    if index is None or index == ':':
        index = slice(None, None, None)

    if not isinstance(index, (slice, str)):
        index = slice(index, (index + 1) or None)

    format = format or filetype(filename, read=isinstance(filename, str))
    io = get_ioformat(format)

    yield from _iread(filename, index, format, io, parallel=parallel,
                      **kwargs)


@parallel_generator
def _iread(filename, index, format, io, parallel=None, full_output=False,
           **kwargs):

    if not io.can_read:
        raise ValueError(f"Can't read from {format}-format")

    if io.single:
        start = index.start
        assert start is None or start == 0 or start == -1
        args = ()
    else:
        args = (index,)

    must_close_fd = False
    if isinstance(filename, str):
        if io.acceptsfd:
            mode = 'rb' if io.isbinary else 'r'
            fd = open_with_compression(filename, mode)
            must_close_fd = True
        else:
            fd = filename
    else:
        assert io.acceptsfd
        fd = filename

    # Make sure fd is closed in case loop doesn't finish:
    try:
        for dct in io.read(fd, *args, **kwargs):
            if not isinstance(dct, dict):
                dct = {'atoms': dct}
            if full_output:
                yield dct
            else:
                yield dct['atoms']
    finally:
        if must_close_fd:
            fd.close()


def parse_filename(filename, index=None, do_not_split_by_at_sign=False):
    if not isinstance(filename, str):
        return filename, index

    basename = os.path.basename(filename)
    if do_not_split_by_at_sign or '@' not in basename:
        return filename, index

    newindex = None
    newfilename, newindex = filename.rsplit('@', 1)

    if isinstance(index, slice):
        return newfilename, index
    try:
        newindex = string2index(newindex)
    except ValueError:
        warnings.warn('Can not parse index for path \n'
                      ' "%s" \nConsider set '
                      'do_not_split_by_at_sign=True \nif '
                      'there is no index.' % filename)
    return newfilename, newindex


def match_magic(data: bytes) -> IOFormat:
    data = data[:PEEK_BYTES]
    for ioformat in ioformats.values():
        if ioformat.match_magic(data):
            return ioformat
    raise UnknownFileTypeError('Cannot guess file type from contents')


def filetype(
        filename: NameOrFile,
        read: bool = True,
        guess: bool = True,
) -> str:
    """Try to guess the type of the file.

    First, special signatures in the filename will be checked for.  If that
    does not identify the file type, then the first 2000 bytes of the file
    will be read and analysed.  Turn off this second part by using
    read=False.

    Can be used from the command-line also::

        $ ase info filename ...
    """

    orig_filename = filename
    if hasattr(filename, 'name'):
        filename = filename.name

    ext = None
    if isinstance(filename, str):
        if os.path.isdir(filename):
            if os.path.basename(os.path.normpath(filename)) == 'states':
                return 'eon'
            return 'bundletrajectory'

        if filename.startswith('postgres'):
            return 'postgresql'

        if filename.startswith('mysql') or filename.startswith('mariadb'):
            return 'mysql'

        # strip any compression extensions that can be read
        root, compression = get_compression(filename)
        basename = os.path.basename(root)

        if '.' in basename:
            ext = os.path.splitext(basename)[1].strip('.').lower()

        for fmt in ioformats.values():
            if fmt.match_name(basename):
                return fmt.name

        if not read:
            if ext is None:
                raise UnknownFileTypeError('Could not guess file type')
            ioformat = extension2format.get(ext)
            if ioformat:
                return ioformat.name

            # askhl: This is strange, we don't know if ext is a format:
            return ext

        if orig_filename == filename:
            fd = open_with_compression(filename, 'rb')
        else:
            fd = orig_filename  # type: ignore[assignment]
    else:
        fd = filename
        if fd is sys.stdin:
            return 'json'

    data = fd.read(PEEK_BYTES)
    if fd is not filename:
        fd.close()
    else:
        fd.seek(0)

    if len(data) == 0:
        raise UnknownFileTypeError('Empty file: ' + filename)

    try:
        return match_magic(data).name
    except UnknownFileTypeError:
        pass

    format = extension2format.get(ext, None)
    if format is not None:
        return format.name

    if guess:
        if ext is not None:
            assert isinstance(ext, str)
            return ext
            # Do quick xyz check:
        lines = data.splitlines()
        if lines and lines[0].strip().isdigit():
            return extension2format['xyz'].name

    raise UnknownFileTypeError('Could not guess file type')


def index2range(index, length):
    """Convert slice or integer to range.

    If index is an integer, range will contain only that integer."""
    obj = range(length)[index]
    if isinstance(obj, numbers.Integral):
        obj = range(obj, obj + 1)
    return obj


# these two will be assigned later (from ase.plugins.__init__)
# to avoid circular import
ioformats: IOFormatPluggables = None     # type: ignore[assignment]
extension2format: ListingView = None    # type: ignore[assignment]
all_formats: IOFormatPluggables = None   # type: ignore[assignment]

# Just here, to avoid circular imports - force load the formats
import ase.plugins as ase_plugins  # NOQA: F401,E402
