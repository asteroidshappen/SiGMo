# What's being tested
import SiGMo as sgm

# Testing
import pytest

# I/O
import os

# Copying
import copy


# pre-define some test parameters
input_param_tuples = [(None, None, None),
                      # (None, None, {'testarg1': 1, 'testarg2': '2a'}),
                      (None, 'tmp_path', None),
                      (None, 'tmp_path', {'testarg1': 1, 'testarg2': '2a'}),
                      # ('Test_Name', None, None),
                      # ('Test_Name', None, {'testarg1': 1, 'testarg2': '2a'}),
                      ('Test_Name', 'tmp_path', None),
                      ('Test_Name', 'tmp_path', {'testarg1': 1, 'testarg2': '2a'})]

load_param_tuples = [(False, False),
                     (False, True),
                     (True, False),
                     (True, True)]

input_and_load_param_tuples = []
for i in input_param_tuples:
    for l in load_param_tuples:
        input_and_load_param_tuples.append((i[0], i[1], i[2], l[0], l[1]))


# pre-define some corresponding helper routine to make and configure each snapshot
def make_and_configure_snp(dirpath, name, tmp_path):
    # minimal instantiation?
    if name is None:
        snp = sgm.Snapshot()
    else:
        snp = sgm.Snapshot(name=name)
    # set the dirpath, set it to tmp_path, or leave at default?
    if dirpath is 'tmp_path':
        snp.dirpath = tmp_path  # setting the temporary path fixture
    elif dirpath is not None:
        snp.dirpath = dirpath  # straight-up setting the path that's entered
    else:
        pass  # do nothing if dirpath is None
    return snp

# +===============================================+
# + A C T U A L   T E S T S   S T A R T   H E R E +
# +===============================================+


# ==============
# save_to_disk()

def test_save_to_disk_minimalinput(tmp_path):
    os.chdir(tmp_path)
    snp = sgm.Snapshot()
    filepath = snp.save_to_disk()
    assert os.path.isfile(filepath)


@pytest.mark.parametrize("name", [(None), (''), ('Foo'), ('Foo_Bar')])
def test_save_to_disk_name_as_basename(name, tmp_path):
    os.chdir(tmp_path)
    snp = sgm.Snapshot(name=name)
    filepath = snp.save_to_disk()
    assert os.path.isfile(filepath)


@pytest.mark.parametrize("name, some_attr", [(None, None), (None, 'stringy McString'), ('Foo', 3), ('Foo_Bar', 3.14159)])
def test_save_to_disk_name_and_attr(name, some_attr, tmp_path):
    os.chdir(tmp_path)
    snp = sgm.Snapshot(name=name, some_attr=some_attr)
    filepath = snp.save_to_disk()
    assert os.path.isfile(filepath)


@pytest.mark.parametrize("basename", [(None), (''), ('Foo'), ('Foo_Bar')])
def test_save_to_disk_metaattr_set_at_creation(basename, tmp_path):
    os.chdir(tmp_path)
    snp = sgm.Snapshot(prefix='Pre', basename=basename, snaptime='20201010050505111111', dirpath=tmp_path)
    filepath = snp.save_to_disk()
    assert os.path.isfile(filepath)


@pytest.mark.parametrize("basename", [(''), ('Foo'), ('Foo_Bar')])
def test_save_to_disk_metaattr_set_at_saving(basename, tmp_path):
    os.chdir(tmp_path)
    snp = sgm.Snapshot()
    combipath = tmp_path / f'Pre_{basename}_20201010050505111111.json'
    filepath = snp.save_to_disk(filepath=combipath)
    assert os.path.isfile(filepath)


@pytest.mark.parametrize("basename", [(''), ('Foo'), ('Foo_Bar')])
def test_save_to_disk_metaattr_reset_at_saving(basename, tmp_path):
    os.chdir(tmp_path)
    snp = sgm.Snapshot(prefix='Ante', basename='Portas', snaptime='44443333222222111111', dirpath=tmp_path)
    combipath = tmp_path / f'Pre_{basename}_20201010050505111111.json'
    filepath = snp.save_to_disk(filepath=combipath)
    assert os.path.isfile(filepath)


# ==================
# update_from_disk()

@pytest.mark.parametrize("warning", [(True), (False)])
def test_update_from_disk_empty_snap(warning, tmp_path):
    os.chdir(tmp_path)
    snp_orig = sgm.Snapshot()
    filepath = snp_orig.save_to_disk()
    # make empty new snapshot and read in from file/update it
    snp_read = sgm.Snapshot()
    snp_read.update_from_disk(filepath=filepath, warning=warning)
    # make sure both are the same (contents preserved after storing to disk
    assert snp_orig == snp_read


@pytest.mark.parametrize("name", [(None), (''), ('foo')])
def test_update_from_disk_other_snap(name, tmp_path):
    os.chdir(tmp_path)
    snp_1 = sgm.Snapshot(name=name, someattr='fudge')
    filepath = snp_1.save_to_disk()
    # make empty new snapshot and read in from file/update it
    snp_2 = sgm.Snapshot(name='bar', someattr='cream', otherattr='liquorice')
    snp_2.update_from_disk(filepath=filepath)
    # make sure both are the same (contents preserved after storing to disk
    assert snp_1 == snp_2


def test_update_from_disk_reset_mangled_snap(tmp_path):
    os.chdir(tmp_path)
    snp_orig = sgm.Snapshot()
    filepath = snp_orig.save_to_disk()
    # mangle copy of the snapshot in memory, then reset by updating from file
    snp_mangled = copy.deepcopy(snp_orig)   # deepcopy important so that the .data dict is not shared, but copied too
    snp_mangled.data['addional_attr'] = 'foo'
    snp_mangled.update_from_disk()
    # make sure both are the same (contents preserved after storing to disk
    assert snp_orig == snp_mangled


# ==================
# load_from_disk()

def test_load_from_disk(tmp_path):
    os.chdir(tmp_path)
    snp_orig = sgm.Snapshot()
    filepath = snp_orig.save_to_disk()
    # make new snapshot based on file straightaway
    snp_new = sgm.Snapshot.load_from_disk(filepath=filepath)
    # make sure both are the same (contents preserved after storing to disk
    assert snp_orig == snp_new
