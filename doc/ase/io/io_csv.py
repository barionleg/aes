# creates: io.csv
from ase.plugins import io_formats

with open('io.csv', 'w') as fd:
    print('format, description, capabilities', file=fd)
    for io in io_formats.sorted:
        c = ''
        if io.can_read:
            c = 'R'
        if io.can_write:
            c += 'W'
        if not io.single:
            c += '+'
        print(f':ref:`{io.name}`, {io.description}, {c}',
              file=fd)
