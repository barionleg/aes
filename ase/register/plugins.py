""" This module contains the classes for listing plugins and
the pluggables (Calculators, IOs, etc...) provided by the plugins.


The structure is as follows


-----------                     --------------------------------
|         |                     |                              |
| Plugins |--------1:n----------|   Plugin (can provide some   |
|         |  (for each plugin   |   calculator, formats...)    |
|         |      package)       |                              |
^^^^^^^^^^                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                           |
   1:n  for each type                         1:n - plugin can return lists
    |   (i.e viewer, calculator)                | of pluggables
    |                                           | of each pluggable type
    |                                           | (calculator, viewer)
    |                                           | that it provides
    |                                           |
    |                                           |
    |                                           |
-------------------------                ------------------------
|                       |                |                      |
|    Pluggables:        |                |    Pluggable:        |
|      list all the     |------1:n-------|    holds inform.     |
|   available 'items'   |                |    about just one    |
|   (calcs, viewers)    |                |    calculator        |
|   of the given type   |                |    or viewer         |
|                       |                |                      |
^^^^^^^^^^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^^^^^^
                                                  |
                                                 1:1
                                                  |
                                         ------------------------
                                         |                      |
                                         |   Implementation     |
                                         |   (e.g. a Calculator |
                                         |    subclass)         |
                                         |                      |
                                         """"""""""""""""""""""""
"""

from contextlib import contextmanager
import importlib
import pkgutil
import warnings
from ase.utils import lazyproperty
from .listing import Listing
_current_plugin = None


@contextmanager
def within_the_plugin(plugin):
    """ When a plugin register() function is called,
    the registered plugin would have to say which plugin
    am I.
    This can lead to errors, so importing of the plugin
    is enclosed by this helper, due to the
    :func:get_currently_registered_plugin
    can say, which plugin is currently imported.
    """

    global _current_plugin
    ocp = _current_plugin
    _current_plugin = plugin
    yield
    _current_plugin = ocp


def get_currently_registered_plugin():
    """
    Which plugin is imported and so to which plugin
    belongs currently imported Pluggables

    See the :func:within_the_plugin
    """
    plugin = _current_plugin
    if not plugin:
        import ase.plugins as ase_plugins
        plugin = ase_plugins.plugins['external']
    return plugin


class Plugins(Listing):
    """ A class, that holds all the installed plugins in
        the given namespace package."""

    """ This information is just for initial creating of the pluggables """
    def __init__(self, namespace_package, pluggable_types):
        self.namespace_package = namespace_package
        self._pluggables = {
            k: cls(k)
            for k, cls in pluggable_types.items()
        }

    def packages(self):
        """ Return all the plugin packages, that are in the
        given namespace package (so that are in 'ase.plugins') """
        package = importlib.import_module(self.namespace_package)

        def mod_name(mod):
            return self.namespace_package + '.' + mod.name

        def import_plugin_module(name, path):
            try:
                module = importlib.import_module(name)
                return module
            except ImportError:
                warnings.warn(f"Can not import {name} in {path}."
                              " This ASE plugin is probably broken.")

        modules = (import_plugin_module(mod_name(mod), mod.module_finder.path)
                   for mod in pkgutil.iter_modules(package.__path__))

        modules = (i for i in modules if i)
        return modules

    def _populate(self):
        self._items = {p.__name__.rsplit('.', 1)[-1]: Plugin(self, p)
                       for p in self.packages()}

    def pluggables_of(self, class_type):
        return self._pluggables[class_type]

    def all_pluggables(self):
        return self._pluggables.values()

    @lazyproperty
    def calculators(self):
        return self.pluggables_of('calculators')

    @lazyproperty
    def viewers(self):
        return self.pluggables_of('viewers')

    @lazyproperty
    def io_formats(self):
        return self.pluggables_of('io_formats')

    def __repr__(self):
        return f"<ASE plugins from: {self.namespace_package}>"

    def info(self, prefix='', opts={}, filter=None):
        return "Plugins:\n"\
               "--------\n" + super().info(prefix, opts, filter)

    def register(self):
        """
        Register all the installed pluggables. To do so
        - import all plugin packages
        - register all the pluggables from the plugins
        """
        self._populate()
        for i in self.values():
            i.register()


class Plugin:
    """ A class, that encapsulates a plugin package """

    def __init__(self, plugins, package):
        self.plugins = plugins
        self.package = package
        self.pluggables = {
            i.class_type: {} for i in plugins.all_pluggables()
        }
        self.modules = {}
        self._pluggables = {}
        self.registered = False

    def add_pluggable(self, pluggable):
        """ Called by Pluggable.register() """
        self.pluggables[pluggable.class_type][pluggable.name] = pluggable

    @property
    def name(self):
        return self.package.__name__[12:]   # get rig 'ase.plugins'

    @property
    def lowercase_names(self):
        return (self.name.lower(), )

    def register(self):
        """ Register the pluggables in the plugin:
        Call the ase_register function from the
        __init__.py of the plugin package
        """
        if not self.registered:
            if hasattr(self.package, 'ase_register'):
                with within_the_plugin(self):
                    self.package.ase_register()
            self.registered = True

    def info(self, prefix='', opts={}):
        info = f'{prefix}{self.name}'

        prefix += '  '
        opts = opts.copy()
        opts['plugin'] = False

        for pluggables in self.plugins.all_pluggables():
            itype = pluggables.class_type
            inst = self.pluggables[itype]
            if inst:
                p = f'\n{prefix}{pluggables.singular_name}: '
                for i in inst.values():
                    info += i.info(p, opts)
        return info

    def __repr__(self):
        return f"<ASE plugin: {self.package}>"
