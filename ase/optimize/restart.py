"""Saving the state of Dynamics objects to a file."""

from ase.io.jsonio import write_json
from ase.utils import IOContext


class JsonDumper(IOContext):

    def __init__(self, dyn, *args):
        self.dyn = dyn
        self.additional_objects = args

        for object in self.additional_objects:
            if not hasattr(object, "todict"):
                raise ValueError(
                    f"Object {object} does not have a todict method.")

    def __del__(self):
        self.close()

    def __call__(self):
        data = self.dyn.todict()

        name = f"{self.dyn.__class__.__name__}.{self.dyn.nsteps}.json"

        with open(name, "w") as f:
            write_json(f, data)

        for object in self.additional_objects:
            with open(
                f"{object.__class__.__name__}.{self.dyn.nsteps}.json", "w"
            ) as f:
                write_json(f, object.todict())
