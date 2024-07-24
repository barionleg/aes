"""
In this plugins, external calculators (either the old long known by
ASE, or the ones registered using entry-points) are present

View plugins can be registered through the entrypoint system in with the
following in a module, such as a `viewer.py` file:

```python3
VIEWER_ENTRYPOINT = ExternalViewer(
    desc="Visualization using <my package>",
    module="my_package.viewer"
)
```

Where module `my_package.viewer` contains a `view_my_viewer` function taking
and `ase.Atoms` object as the first argument, and also `**kwargs`.

Then ones needs to register an entry point in `pyproject.toml` with

```toml
[project.entry-points."ase.visualize"]
my_viewer = "my_package.viewer:VIEWER_ENTRYPOINT"
```

After this, call to `ase.visualize.view(atoms, viewer='my_viewer')` will be
forwarded to `my_package.viewer.view_my_viewer` function.
"""

import warnings
from ase.register import register_calculator
from ase.io.formats import define_io_format
from ase.visualize.viewers import define_viewer
from ase.utils.plugins import ExternalIOFormat
from ase.register.plugins import get_currently_registered_plugin
from ase.utils.plugins import ExternalViewer
from importlib.metadata import entry_points


def define_external_io_format(entry_point):

    fmt = entry_point.load()
    # if entry_point.name in ioformats:
    #    raise ValueError(f'Format {entry_point.name} already defined')
    if not isinstance(fmt, ExternalIOFormat):
        raise TypeError('Wrong type for registering external IO formats '
                        f'in format {entry_point.name}, expected '
                        'ExternalIOFormat')
    fmt = define_io_format(entry_point.name, **fmt._asdict(), external=True)
    fmt.plugin = get_currently_registered_plugin()
    fmt.register()


def register_external_io_formats(group):
    if hasattr(entry_points(), 'select'):
        fmt_entry_points = entry_points().select(group=group)
    else:
        fmt_entry_points = entry_points().get(group, ())

    for entry_point in fmt_entry_points:
        try:
            define_external_io_format(entry_point)
        except Exception as exc:
            warnings.warn(
                'Failed to register external '
                f'IO format {entry_point.name}: {exc}'
            )


def define_external_viewer(entry_point):
    """Define external viewer"""

    viewer_def = entry_point.load()
    # if entry_point.name in VIEWERS:
    #    raise ValueError(f"Format {entry_point.name} already defined")
    if not isinstance(viewer_def, ExternalViewer):
        raise TypeError(
            "Wrong type for registering external IO formats "
            f"in format {entry_point.name}, expected "
            "ExternalViewer"
        )
    define_viewer(entry_point.name, **viewer_def._asdict(),
                  external=True)


def register_external_viewer_formats(group):
    if hasattr(entry_points(), "select"):
        viewer_entry_points = entry_points().select(group=group)
    else:
        viewer_entry_points = entry_points().get(group, ())

    for entry_point in viewer_entry_points:
        try:
            define_external_viewer(entry_point)
        except Exception as exc:
            warnings.warn(
                "Failed to register external "
                f"Viewer {entry_point.name}: {exc}"
            )


def ase_register():
    register_calculator("asap3.EMT")
    register_calculator("gpaw.GPAW")
    register_calculator("hotbit.Calculator", name='hotbit')

    # Register IO formats exposed through the ase.ioformats entry point
    register_external_io_formats('ase.ioformats')
    register_external_viewer_formats("ase.visualize")
