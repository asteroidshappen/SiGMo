# general
import numpy as np
import warnings
from tqdm import tqdm

# I/O
import json
import os

# date and time
from datetime import datetime


# Defining additional helper methods
def join_paths(path1, path2):
    """Join two paths or a path and a filename, assuming that if they're not strings, then they're path objects"""
    if (type(path1) is str) and (type(path2) is str):
        combipath = path1 + path2
    else:
        combipath = path1 / path2
    return combipath

def assign_with_warning(target, value, warning: bool = True):
    """Simple assigning of value to target that warns if value not 'equal' ot target"""
    if (value != target) and warning:
        print(f"WARNING! Not identical, {target!r} replaced by {value!r}")
    return value

def split_path_and_name(inpath, sep='_'):
    """Split filename and path and re-contruct meta-attributes from it"""
    dirpath, filename = os.path.split(inpath)

    meta_attr_strings = filename.split(sep)
    prefix = meta_attr_strings[0]
    basename = '_'.join(meta_attr_strings[1:-1])
    *snaptime, extension = meta_attr_strings[-1].split('.')
    return dirpath, filename, prefix, basename, snaptime, extension

def read_all_snapshots_from_dir(
        snp_dir,
        n_envs=None, n_halos=None, n_gals=None,
        single_snapshots=False, n_steps=None,
        fn_start = "snp",  # string elements to split and recognise file names by
        fn_middle_gal = "Galaxy",
        fn_middle_halo = "Halo",
        fn_middle_env = "Environment",
        fn_end = "",
        fn_separator = "_"
):
    """
    Reads in all snapshot files from a specified directory

    :param snp_dir: Input directory
    :param n_envs: Number of Environment objects at any point in time during the simulation (usually: 1)
    :param n_halos: Number of Halo objects at any point in time during the simulation
    :param n_gals: Number of Galaxy objects at any point in time during the simulation
    :param single_snapshots: Is the input individual snapshots for each object and time step (True),
        or one per Environment and time step (False; default)
    :param n_steps: Number of time steps in the simulation (or simulation output if wtd > 1)
    :param fn_start: String element used to identify the beginning of snapshot file names
    :param fn_middle_gal: String element used to identify the middle part of Galaxy snapshot file names
    :param fn_middle_halo: String element used to identify the middle part of Halo snapshot file names
    :param fn_middle_env: String element used to identify the middle part of Environment snapshot file names
    :param fn_end: String element used to identify the end part of snapshot files (can be left empty)
    :param fn_separator: The separator used to split the full file names in order to analyse/compare them
    :return: env_grid, halo_grid, gal_grid: numpy arrays with the snapshots of all Environments,
        Halos and Galaxies over all time steps of the simulation. Example with 2000 time steps, one Environment
        and 379 Halos in it with one Galaxy each:
        env_grid.shape=(1, 2000), halo_grid.shape=(379, 2000), gal_grid.shape=(379, 2000)
    """
    # get all file names
    print(f"Reading all snapshots from directory '{snp_dir}'")
    file_iter = snp_dir.iterdir()
    file_list = list(file_iter)
    n_files = len(file_list)

    # how many timesteps?
    if single_snapshots:
        # n_steps = 800
        # n_steps = 799
        if n_steps is None:
            raise TypeError("'n_steps' has not been set, but is required for 'single_snapshots=True' mode")
    else:
        n_steps = n_files

    # How many envs, halos, gals if numbers not provided?
    # if number of halos and galaxies is not (properly) provided
    # WARNING::: this HARDCODES 1 Environment total and same number of Galaxies as Halos (one each)!
    n_envs = n_envs if n_envs else 1
    if (not n_halos) or (not n_gals):
        # find an Environment snapshot file
        _tmp_filepath = None
        for _tmp_filepath in file_list:
            if _tmp_filepath.resolve().name.split(fn_separator)[1] == fn_middle_env:
                _tmp_snp = Snapshot.load_from_disk(_tmp_filepath)
                break
        # set number of Halos
        if not n_halos:
            # catch the case where the above didn't find an Environment snapshot file
            try:
                n_halos = len(_tmp_snp.data["halos"])
            except AttributeError:
                print(f"'n_halos' could not be set, because no Snapshot file with "
                      f"'{fn_middle_env}' in it/in the right place could be found "
                      f"in directory '{snp_dir}'")
        # set number of Galaxies
        if not n_gals:
            n_gals = n_halos

    # make arrays here already
    env_grid = np.empty(shape=(n_envs, n_steps), dtype=object)
    halo_grid = np.empty(shape=(n_halos, n_steps), dtype=object)
    gal_grid = np.empty(shape=(n_gals, n_steps), dtype=object)

    # loop trough the files in the arbitrary order that they got in the iterator file_iter
    for file in tqdm(file_list, total=n_files):
        fn = file.name

        # check that fn is regular snapshot of time evolution, not other file (e.g. "final" snapshot etc)
        if fn.startswith(fn_start):
            # split the file name (not the whole path)
            fn_split = fn.split(fn_separator)

            # assign the different elements from file name to different counters (time and object/param combination)
            t = int(fn_split[0][len(fn_start):])
            i = int(fn_split[2]) if len(fn_split) == 4 else 0
            snp_type = fn_split[1]

            # make full file path with directories
            full_file_path = file.resolve()

            # differentiate between Galaxy, Halo and Environment files
            if snp_type == fn_middle_env:
                env_grid[..., t].flat[i] = Snapshot.load_from_disk(full_file_path)
            elif snp_type == fn_middle_halo:
                if single_snapshots:
                    halo_grid[..., t].flat[i] = Snapshot.load_from_disk(full_file_path)
                else:
                    warnings.warn(f"The type '{snp_type}' of snapshot '{file}' is not yet supported for multi_snapshot "
                                  f"read in!\nSnapshot was not read in.")
            elif snp_type == fn_middle_gal:
                if single_snapshots:
                    gal_grid[..., t].flat[i] = Snapshot.load_from_disk(full_file_path)
                else:
                    warnings.warn(f"The type '{snp_type}' of snapshot '{file}' is not yet supported for multi_snapshot "
                                  f"read in!\nSnapshot was not read in.")
            else:
                warnings.warn(
                    f"The type '{snp_type}' of snapshot '{file}' was not recognised! Snapshot was not read in.")

    return env_grid, halo_grid, gal_grid


# Snapshot Class: custom class with a dict that contains all of one AstroObject's attributes
class Snapshot:

    def __init__(self, *args, **kwargs):
        # set default values that need to be determined at runtime
        self.prefix = 'snp'
        self.basename = kwargs['name'] if ('name' in kwargs.keys() and kwargs['name'] is not None) else ''
        self.snaptime = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.dirpath = ''

        # if existent, unpack meta-attributes from kwargs and set as object attributes, else keep defaults
        meta_attr_defaults = {'prefix': self.prefix,
                              'basename': self.basename,
                              'snaptime': self.snaptime,
                              'dirpath': self.dirpath}
        for meta_attr_key, meta_attr_value in meta_attr_defaults.items():
            if meta_attr_key in kwargs.keys():
                newvalue = kwargs.pop(meta_attr_key)
                newvalue = meta_attr_value if newvalue is None else newvalue    # check that new value is not None
                setattr(self, meta_attr_key, newvalue)

        # use the args and (remaining) kwargs to set dictionary with data values
        self.data = dict(*args, **kwargs)

        self.filename = self.prefix + "_" + self.basename + "_" + self.snaptime + ".json"
        return

    def __repr__(self):
        """Representation of the object. Concise overview, object cannot be created by running eval on this"""
        r_string = ", ".join("=".join((str(k), repr(v))) for k, v in vars(self).items())
        return f"{type(self).__name__}({r_string})"

    def __eq__(self, other):
        """Check whether contents (data and meta-attributes) are the same, using self.__dict__"""
        return True if self.__dict__ == other.__dict__ else False

    def save_to_disk(self, filepath: str = None):
        """Save snapshot's data dict. to json file. Additional attributes used to make filename/path if not included"""
        # concat dirpath and filename depending on whether they're strings or not (assumed path objects)
        filepath = join_paths(self.dirpath, self.filename) if filepath is None else filepath

        # write the file
        with open(filepath, 'w') as outfile:
            json.dump(self.data, outfile, default=vars, indent=2)
        return filepath

    def update_from_disk(self, filepath: str = None, warning=True):
        """Load a snapshot from json file on disk and update current object. The filepath needs to be passed."""
        # concat dirpath and filename depending on whether they're strings or not (assumed path objects)
        filepath = join_paths(self.dirpath, self.filename) if filepath is None else filepath

        # read the file and storing the input (dict format) in data attribute
        with open(filepath, 'r') as infile:
            self.data = json.load(infile)

        # split filename and path and re-contruct meta-attributes from it
        dirpath, filename, prefix, basename, snaptime, extension = split_path_and_name(filepath)

        # save re-constructed attr in obj
        self.dirpath = assign_with_warning(self.dirpath, dirpath, warning)
        self.filename = assign_with_warning(self.filename, filename, warning)
        self.prefix = assign_with_warning(self.prefix, prefix, warning)
        self.basename = assign_with_warning(self.basename, basename, warning)
        self.snaptime = assign_with_warning(self.snaptime, snaptime, warning)
        return self.data

    @classmethod
    def load_from_disk(cls, filepath: str):
        """Create new Snapshot object from data of one stored on disk. The file needs to be passed"""
        # instantiate Snapshot object, but empty
        snp = cls()

        # use update_from_disk() to fill the dict part up, suppress warnings about differences in meta-params (expected)
        data = snp.update_from_disk(filepath, warning=False)
        return snp

    def autoname_with_index(self, index, index_max):
        index_maxlen = len(str(index_max))
        name = self.data['name']
        z = self.data['z']
        return f'snp{index:0{index_maxlen}}_{name}_z{z:.{index_maxlen}f}.json'
