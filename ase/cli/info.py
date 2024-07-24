# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution

from itertools import product
from typing import Optional, List


class CLICommand:
    """Print information about files or system.

    Without arguments, show information about ASE installation
    and library versions of dependencies.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--files', nargs='*', metavar='PATH',
                            help='Print information about specified files.')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show more information about files.')
        parser.add_argument('--config', action='store_true',
                            help='List configured calculators')
        parser.add_argument('--formats', nargs='*', metavar='NAME',
                            dest='io_formats',
                            help='List file formats: all known to ASE, or the '
                                 'ones with the given string(s) in the name')
        parser.add_argument('--calculators', nargs='*', metavar='NAME',
                            help='List calculators and their configuration:'
                                 ' all known to ASE, or the ones with the given'
                                 ' string(s) in the name')
        parser.add_argument('--viewers', nargs='*', metavar='NAME',
                            help='List viewers: all the known to ASE, or the '
                                 'ones with the given string(s) in the name')
        parser.add_argument('--plugins', nargs='*', metavar='NAME',
                            help='List plugins: all installed, or the ones '
                                 'with the given string(s) in the name')

    @staticmethod
    def run(args):
        print_info()
        if args.files:
            print_file_info(args)

        if args.config:
            print()
            from ase.config import cfg
            cfg.print_everything()

        for kind in ('io_formats', 'calculators', 'viewers', 'plugins'):
            val = getattr(args, kind)
            if val is None:
                continue
            print()
            print_pluggables(kind, val or None)


def print_file_info(args):
    from ase.io.formats import UnknownFileTypeError, filetype, ioformats
    from ase.io.bundletrajectory import print_bundletrajectory_info
    from ase.io.ulm import print_ulm_info
    n = max(len(filename) for filename in args.files) + 2
    nfiles_not_found = 0
    for filename in args.files:
        try:
            format = filetype(filename)
        except FileNotFoundError:
            format = '?'
            description = 'No such file'
            nfiles_not_found += 1
        except UnknownFileTypeError:
            format = '?'
            description = '?'
        else:
            if format in ioformats:
                description = ioformats[format].description
            else:
                description = '?'

        print('{:{}}{} ({})'.format(filename + ':', n,
                                    description, format))
        if args.verbose:
            if format == 'traj':
                print_ulm_info(filename)
            elif format == 'bundletrajectory':
                print_bundletrajectory_info(filename)

    raise SystemExit(nfiles_not_found)


def print_info():
    import platform
    import sys

    from ase.dependencies import all_dependencies

    versions = [('platform', platform.platform()),
                ('python-' + sys.version.split()[0], sys.executable)]

    for name, path in versions + all_dependencies():
        print(f'{name:24} {path}')


def print_pluggables(kind: str, allowed_names: Optional[List[str]] = None):
    import ase.plugins as plugins
    to_print = getattr(plugins, kind)
    if allowed_names is not None:
        allowed_names = [kind.lower() for kind in allowed_names]

        def filter(pluggable):
            return any(pattern in name
                       for (pattern, name) in
                       product(allowed_names, pluggable.lowercase_names)
                       )
    else:
        filter = None  # type: ignore[assignment]

    print(to_print.info(filter=filter))
