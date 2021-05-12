import io
import pytest

from ase import Atoms
from ase.db import connect
from ase.db.web import Session
from ase.db.app import DBApp, request2string
from ase.io import read


@pytest.fixture(scope='module')
def database(tmp_path_factory):
    with tmp_path_factory.mktemp('dbtest') as dbtest:
        db = connect(dbtest / 'test.db', append=False)
        x = [0, 1, 2]
        t1 = [1, 2, 0]
        t2 = [[2, 3], [1, 1], [1, 0]]

        atoms = Atoms('H2O',
                      [(0, 0, 0),
                       (2, 0, 0),
                       (1, 1, 0)])
        atoms.center(vacuum=5)
        atoms.set_pbc(True)

        db.write(atoms,
                 foo=42.0,
                 bar='abc',
                 data={'x': x,
                       't1': t1,
                       't2': t2})
        db.write(atoms)

        yield db


@pytest.fixture(scope='module')
def client(database):
    pytest.importorskip('flask')

    app = DBApp()
    app.add_project(database)
    app.flask.testing = True
    return app.flask.test_client()


def test_favicon(client):
    assert client.get('/favicon.ico').status_code == 308  # redirect
    assert client.get('/favicon.ico/').status_code == 204  # no content


def test_db_web(client):
    page = client.get('/').data.decode()
    sid = Session.next_id - 1
    assert 'foo' in page
    for url in [f'/update/{sid}/query/bla/?query=id=1',
                '/default/row/1']:
        resp = client.get(url)
        assert resp.status_code == 200

    for type in ['json', 'xyz', 'cif']:
        url = f'atoms/default/1/{type}'
        resp = client.get(url)
        assert resp.status_code == 200
        atoms = read(io.StringIO(resp.data.decode()), format=type)
        print(atoms.numbers)
        assert (atoms.numbers == [1, 1, 8]).all()


@pytest.fixture
def dbsetup(database):
    pytest.importorskip('flask')

    class DBSetup:
        def __init__(self):
            self.session = Session('name')
            self.project = {'default_columns': ['bar'],
                            'handle_query_function': request2string}
            self.session.update('query', '', {'query': ''}, self.project)
            self.table = self.session.create_table(database, 'id', ['foo'])

    return DBSetup()


def test_add_columns(database, dbsetup):
    """Test that all keys can be added also for row withous keys."""
    table = dbsetup.table
    table = dbsetup.session.create_table(database, 'id', ['foo'])
    assert table.columns == ['bar']  # selected row doesn't have a foo key
    assert 'foo' in table.addcolumns  # ... but we can add it


def test_paging(database, dbsetup):
    """Test paging."""
    assert len(dbsetup.table.rows) == 2

    session = dbsetup.session
    session.update('limit', '1', {}, dbsetup.project)
    session.update('page', '1', {}, dbsetup.project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 1

    # We are now on page 2 and select something on page 1:
    session.update('query', '', {'query': 'id=1'}, dbsetup.project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 1
