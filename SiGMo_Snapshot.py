# I/O
import json
import os

# date and time
from datetime import datetime

# Snapshot Class: just a dict with its own class name for easy identification
class Snapshot(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basename = self['name'] if 'name' in self.keys() else ''
        self.snaptime = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.filename = "snp_" + self.basename + "_" + self.snaptime + ".json"
        self.dirpath = ""
        return

    def save_to_disk(self, outpath: str = None):
        # concat dirpath and filename depending on whether they're strings or not (assumed path objects)
        if outpath is None:
            if (type(self.dirpath) is str) and (type(self.filename) is str):
                outpath = self.dirpath + self.filename
            else:
                outpath = self.dirpath / self.filename
        # write the file
        with open(outpath, 'w') as outfile:
            json.dump(self, outfile)
        return outpath

    def load_from_disk(self, inpath: str = None):
        # concat dirpath and filename depending on whether they're strings or not (assumed path objects)
        if inpath is None:
            if (type(self.dirpath) is str) and (type(self.filename is str)):
                inpath = self.dirpath + self.filename
            else:
                inpath = self.dirpath / self.filename
        # read the file
        with open(inpath, 'r') as infile:
            tmp = json.load(infile)
        # setting the read values as dict-like entries in this Snapshot instance
        for k, v in tmp.items():
            self[k] = v
        return tmp

    @classmethod
    def from_file(cls, filepath: str):
        # use filepath and filename info to have some later attributes ahead of time
        dirpath, filename = os.path.split(filepath)
        basename = filename[4:-26]
        snaptime = filename[-25:-5]

        # instantiate Snapshot object, but empty
        snp = cls()

        # fill with the pre-determined attributes
        snp.basename = basename
        snp.snaptime = snaptime
        snp.filename = filename
        snp.dirpath = dirpath

        # use load_from_disk() to fill the dict part up
        tmp = snp.load_from_disk(filepath)

        return snp