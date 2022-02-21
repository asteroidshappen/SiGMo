# I/O (ujson much faster than regular json, but orjson should be even faster but more complicated to install)
import json
# try:
#     import ujson as json
# except ImportError:
#     import json

import os

# date and time
from datetime import datetime

# # Making copies
# import copy

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


# Snapshot Class: just a dict with its own class name for easy identification
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

    # def __copy__(self):
    #     """Shallow copy implemented using the Lib/copy.py module"""
    #     return copy.copy(self)

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
