# Contains class to create sets of fully spelled-out initial conditions

# ======
# Import

# basic
import copy
import warnings
import numpy as np

import SiGMo


# =====================================================================================
# Class for all ICs, with different constructors for different initialisation scenarios

class IC:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values
        return

    @classmethod
    def single_param(cls, key, values):
        keys = {key: 0}     # position in values array along axis 0 is 0, since it's the only parameter at the moment
        values = np.array(values)
        values.shape = (len(values), len(keys))
        return cls(keys, values)

    def __add__(self, other):
        """The addition for two ICs is to stick the other IC onto the self IC, element-wise.
        Therefore, shapes must match.

        Example:

        self.keys: {a: 0, b: 1}
        other.keys: {m: 0}
        with a, b and m being different parameters like stellar mass, star-formation rate and halo mass

        There are in this example 3 different parameter configurations (1, 2, 3), so the values would look like this:

        self.values: [[a1, b1], [a2, b2], [a3, b3]]
        other.values: [[m1], [m2], [m3]]

        Then if you add self and other, this should be the result:
        (self + other).keys: {a: 0, b: 1, m: 2}
        (self + other).values: [[a1, b1, m1], [a2, b2, m2], [a3, b3, m3]]

        In array terminology, self.values.shape[0] == other.values.shape[0] must be True
        """


        # make new instance of IC object that will be returned, based on self (aka the first operand)
        IC_add = copy.deepcopy(self)

        # adding together
        if len(self) == len(other):     # check number of param combinations is the same
            duplicate_l = []
            for k, i in other.keys.items():
                if k in IC_add.keys:
                    # replace the previous values if entry already existed
                    duplicate_l.append([k, IC_add.keys[k]])
                    IC_add.values[:,IC_add.keys[k]] = other.values[:,i]
                else:
                    # plain add new entries of they didn't exist before
                    IC_add.values = np.append(IC_add.values, other.values[:,i].reshape(-1, 1), axis=1)
                    IC_add.keys[k] = len(IC_add.keys)
            if len(duplicate_l) > 0:
                warnings.warn(f"While combining two ICs, values for the following keys were replaced: "
                              f"{', '.join(ki[0] for ki in duplicate_l)}")
        else:
            raise ValueError(f"ICs could not be added due to different lengths of {len(self)} and {len(other)}")

        return IC_add

    def __sub__(self, other):
        """The subtraction for two ICs is to remove the other IC from the self IC, element-wise.
        Therefore, shapes must match.

        Example:

        self.keys: {a: 0, b: 1, c: 2}
        other.keys: {c: 0}
        with a, b and c being different parameters like stellar mass, star-formation rate and halo mass

        There are in this example 4 different parameter configurations (1, 2, 3, 4), so the values would look like this:

        self.values: [[a1, b1, c1], [a2, b2, c1], [a3, b3, c1], [a4, b4, c4]]
        other.values: [[c1], [c2], [c3], [c4]]

        Then if you subtract other from self, this should be the result:
        (self - other).keys: {a: 0, b: 1}
        (self + other).values: [[a1, b1], [a2, b2], [a3, b3], [a4, b4]

        In array terminology, self.values.shape[0] == other.values.shape[0] must be True
        """
        # make new instance of IC object that will be returned, based on self (aka the first operand)
        IC_sub = copy.deepcopy(self)

        # subtracting other from self
        if len(self) == len(other):     # check number of param combinations is the same
            not_in_self_l = []
            for k, i in other.keys.items():
                if k in IC_sub.keys:
                    # remove entry from the resulting IC's values array and keys dict
                    IC_sub.values = np.delete(IC_sub.values, IC_sub.keys[k], axis=1)
                    i_pop = IC_sub.keys.pop(k)
                    # adjust all indexes stored in the resulting IC's keys dict that are larger/after the deleted one
                    for k_rem, i_rem in IC_sub.keys.items():
                        if i_rem > i_pop:
                            IC_sub.keys[k_rem] -= 1

                else:
                    # track non-existing entries in self for the warning message
                    not_in_self_l.append([k, i])

            if len(not_in_self_l) > 0:
                warnings.warn(f"While removing one IC from the other, values for the following keys were not found "
                              f"in the original IC: {', '.join(ki[0] for ki in not_in_self_l)}")
        else:
            raise ValueError(f"ICs could not be subtracted due to different lengths of {len(self)} and {len(other)}")

        return IC_sub

    def __mul__(self, other):
        """The multiplication for two ICs is to expand the self IC grid by other IC.
        The shapes do not need to match.

        Example:

        self.keys: {a, b, c}
        other.keys: {m, n}
        with a, b and m, n being different parameters like stellar mass, star-formation rate and halo mass

        There are in this example 4 different parameter configurations (1, 2, 3, 4) in self, and 3 different
        configurations in other, so the values would look like this:

        self.values: [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3], [a4, b4, c4]]
        other.values: [[m1, n1], [m2, n2], [m3, n3]]

        Then if you multiply self and other, this should be the result:
        (self * other).keys: {a: 0, b: 1, c: 2, m: 3, b4}
        (self * other).values: [[a1, b1, c1, m1, n1], [a2, b2, c2, m1, n1], [a3, b3, c3, m1, n1], [a4, b4, c4, m1, n1],
                                [a1, b1, c1, m2, n2], [a2, b2, c2, m2, n2], [a3, b3, c3, m2, n2], [a4, b4, c4, m2, n2],
                                [a1, b1, c1, m3, n3], [a2, b2, c2, m3, n3], [a3, b3, c3, m3, n3], [a4, b4, c4, m3, n3]]

        The resulting array has the same number of dimensions as the original arrays.

        [ N O T   I M P L E M E N T E D   Y E T !]

        """
        raise FutureWarning("Multiplication of ICs had not yet been implemented.")
        return

    def __truediv__(self, other):
        """The division for two ICs is to reduce the self IC grid by the other IC.
        The shapes do not need to match, but other must be broadcastable to self. Example:

        [ N O T   I M P L E M E N T E D   Y E T !]

        """
        raise FutureWarning("Division of ICs had not yet been implemented.")
        return

    def __len__(self):
        """The length of an IC object is the number of different parameter configurations;
        this corresponds to self.values.shape[0]"""
        return self.values.shape[0] if len(self.keys) >= 1 else len(self.keys)

    def __getitem__(self, item):
        """
        The getitem dunder method returns the indexed parameter configuration as a dict. This enables direct
        use for AstroObject creation through ** unpacking.

        :param item: index of one set of IC with one values per parameter
        :return: dict of the parameter set for the one IC selected by its index, item
        """
        return {k: self.values[item, v] for k, v in self.keys.items()}

    def shrink(self):
        pass

    def allvalues(self, key):
        """
        Returns all values one or several parameter(s), designated by key, take(s) according to self.values

        :param key: string or list/tuple of strings corresponding to a/several key/s in the self.keys dictionary
        :return: an array (for multiple keys: tuple of arrays) of all the values key takes in self.values
        :raises KeyError: if the key/one of the keys is not found in self.dict
        :raises TypeError: if the type of key is neither str nor list/tuple of str
        """
        if bool(key) and isinstance(key, str):
            try:
                values = self.values[:, self.keys[key]]
            except KeyError:
                raise KeyError(f"The key {key!r} has not been found in self.keys, and the corresponding values were "
                               f"not retrieved")
        elif bool(key) and isinstance(key, (list, tuple)) and all(isinstance(elem, str) for elem in key):
            try:
                values = tuple(self.values[:, self.keys[onekey]] for onekey in key)
            except KeyError:
                raise KeyError(f"One ore more keys in {key!r} has not been found in self.keys, and the corresponding "
                               f"values were not retrieved")
        else:
            raise TypeError(f"The type of key {key!r} is not acceptable for look-up in self.keys")
        return values


def create_AstroObject_Ensemble(
        name_env,
        IC_env,
        name_halos,
        IC_halos,
        name_gals,
        IC_gals
):
    """
    Helper routine that creates 1 Environment, n Halo and n Galaxy objects.
    Multiple Galaxies per Halo are not supported.

    :param IC_env: IC of length 1 for the single Environment to be created (so only one set of parameters)
    :param IC_halos: IC of length n for the arbitrary number of Halos to be created in this Environment
    :param IC_gals: IC of SAME length n as the IC for the Halos, to create ONE Galaxy in each of the n Halos.
    :return: env: the Environment object created from IC_env
        halo_arr: a numpy array of the Halo objects created from IC_halos
        gal_arr: a numpy array of the Galaxy objects created from IC_gals
    """
    # pre-allocate arrays for the different halos and galaxies about to be created
    halo_arr = np.empty(shape=(len(IC_halos),), dtype=object)
    gal_arr = np.empty(shape=(len(IC_gals),), dtype=object)

    # create the Environment as well as different Halos and Galaxies
    env = SiGMo.Environment(name=name_env, **(IC_env[0]))
    for i, (name_h, IC_h) in enumerate(zip(name_halos, IC_halos)):
        halo_arr[i] = env.create_Halo(name=name_h, **IC_h)
    for i, (name_g, IC_g) in enumerate(zip(name_gals, IC_gals)):
        gal_arr[i] = halo_arr[i].create_Galaxy(name=name_g, **IC_g)
    return env, halo_arr, gal_arr
