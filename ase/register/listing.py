from collections.abc import Mapping
from ase.utils import lazyproperty
from typing import Dict, Optional, Callable
from collections import defaultdict


def _item_attribute(obj, attribute):
    """ Return item attribute. If it is a list (or set), yield
    all the list items.
    """
    if hasattr(obj, attribute):
        v = getattr(obj, attribute)
        if isinstance(v, (list, set)):
            for i in v:
                yield i
        else:
            yield v


def _item_has_attribute(obj, attribute, value) -> bool:
    for i in _item_attribute(obj, attribute):
        if i == value:
            return True
    return False


class ListingView(Mapping):
    """ View, that lists items in a given listing according an
    given attribute (e.g. extensions of an IO_Format) """

    def __init__(self, listing, attribute, mapping=None):
        self.listing = listing
        self.attribute = attribute
        self.mapping = mapping or (lambda x: x)

    @lazyproperty
    def map(self):
        """ Return the followin mapping:
          <given attribute>:[list of objects having the attribute] """
        out = defaultdict(lambda: [])
        for item in self.listing.values():
            for attr in _item_attribute(item, self.attribute):
                out[attr].append(self.mapping(item))
        out.default_factory = None
        return out

    def __getitem__(self, name):
        return self.map[name][0]

    def __iter__(self):
        return iter(self.map)

    def get_items(self, name):
        """ Return a list of all the items having the attribute """
        return self.map.get(name)

    def __len__(self):
        return len(self.map)


class BaseListing(Mapping):
    """ Class, that lists something, e.g. Plugins or Pluggables
    (of calculators or formats etc...).
    The added items are required to have a name attribute.

    The difference against dict is, that Listing iterates the
    values by default, use keys() to iterate the keys.
    """

    def add(self, item):
        """ Add an item """
        self._items[item.name] = item

    def info(self, prefix: str = '', opts: Dict = {},
             filter: Optional[Callable[[object], bool]] = None) -> str:
        """
        Parameters
        ----------
        prefix
            Prefix, which should be prepended before each line.
            E.g. indentation.
        opts
            Dictionary, that can holds options,
            what info to print and which not.

        Returns
        -------
        info
          Information about the object and (if applicable) contained items.
        """
        frm = self.sorted
        if filter:
            frm = (i for i in frm if filter(i))
        out = [i.info(prefix) for i in frm]
        if filter and not out:
            out = [prefix + ' ...sorry, no such item found']
        return '  \n'.join(out)

    def filter(self, filter):
        """ Return a mapping with only the items thas pass through
        the filter """
        return LazyListing(lambda: {k: v for k, v in self._items.items()
                                    if filter(v)})

    @staticmethod
    def _sorting_key(i):
        return i.name.lower()

    def items(self):
        return self._items.items()

    @lazyproperty
    def sorted(self):
        """ Return items in the listing, sorted by a predefined criteria """
        ins = list(self._items.values())
        ins.sort(key=self._sorting_key)
        return ins

    def __len__(self):
        return len(self._items)

    def __getitem__(self, name):
        out = self.find_by_name(name)
        if not out:
            raise KeyError(f"There is no {name} in {self}")
        return out

    def __iter__(self):
        return iter(self._items)

    def values(self):
        return self._items.values()

    def keys(self):
        return self._items.keys()

    def find_by(self, attribute, value):
        """ Find plugin according the given attribute.
        The attribute can be given by list of alternative values,
        or not at all - in this case, the default value for the attribute
        will be used """
        for i in self.values():
            if _item_has_attribute(i, attribute, value):
                return i

    def find_all_by(self, attribute, value):
        """ Find plugin according the given attribute.
        The attribute can be given by list of alternative values,
        or not at all - in this case, the default value for the attribute
        will be used """
        return (i for i in self.values() if
                _item_has_attribute(i, attribute, value))

    def find_by_name(self, name):
        return self._items.get(name, None)

    def view_by(self, name, mapping=None) -> ListingView:
        return ListingView(self, name, mapping)


class Listing(BaseListing):
    """ Listing holds its own datas """
    def __init__(self):
        self._items = {}


class LazyListing(BaseListing):

    def __init__(self, lazy):
        self._lazy = lazy

    @lazyproperty
    def _items(self):
        return self._lazy()
