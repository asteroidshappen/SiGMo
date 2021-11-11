# What's being tested
import SiGMo_Snapshot as sgm_snp

# Testing
import pytest

# I/O
import os


# pre-define some test parameters
input_param_tuples = [(None, None, None),
                      (None, None, {'testarg1': 1, 'testarg2': '2a'}),
                      (None, 'tmp_path', None),
                      (None, 'tmp_path', {'testarg1': 1, 'testarg2': '2a'}),
                      ('Test_Name', None, None),
                      ('Test_Name', None, {'testarg1': 1, 'testarg2': '2a'}),
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
        snp = sgm_snp.Snapshot()
    else:
        snp = sgm_snp.Snapshot(name=name)
    # set the dirpath, set it to tmp_path, or leave at default?
    if dirpath is 'tmp_path':
        snp.dirpath = tmp_path  # setting the temporary path fixture
    elif dirpath is not None:
        snp.dirpath = dirpath  # straight-up setting the path that's entered
    else:
        pass  # do nothing if dirpath is None
    return snp

# ==========================
# actual testing starts here

@pytest.mark.parametrize("name, dirpath, moreargs", input_param_tuples)
def test_save_to_disk(name, dirpath, moreargs, tmp_path):

    # make and configure the snapshot
    snp = make_and_configure_snp(dirpath, name, tmp_path)

    # actually creating the file, then assert that it exists
    filepath = snp.save_to_disk()
    assert os.path.isfile(filepath)


@pytest.mark.parametrize("name, dirpath, moreargs, newsnap, inpathsupplied", input_and_load_param_tuples)
def test_load_from_disk(name, dirpath, moreargs, newsnap, inpathsupplied, tmp_path):

    # make and configure the snapshot
    snp_orig = make_and_configure_snp(dirpath, name, tmp_path)

    # creating file from snapshot
    filepath = snp_orig.save_to_disk()

    # define where the file will be read to
    if newsnap:
        # make empty snapshot and read original snapshot back in from file
        snp_read = sgm_snp.Snapshot()
    else:
        # keep the original snapshot
        snp_read = snp_orig.copy()  # OPEN QUESTION: should the dict-like contents of snp_orig be cleared/overwritten with None?

    # read in the file
    if inpathsupplied:
        snp_read.load_from_disk(filepath)
    else:
        snp_read.load_from_disk()

    assert snp_read == snp_orig  # I'M PREEEEETTY SURE THIS COMPARISION IS WRONG!! (best case might just compare dict-like part of the snapshot)


def test_from_file():
    assert False
