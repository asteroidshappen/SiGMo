# =======
# Imports

# debug for use in other environments
import platform
print("python version that executes SiGMo is", platform.python_version())

# basic
import copy
import inspect
import warnings
from pathlib import Path

import numpy as np

# OOP
from abc import ABC, abstractmethod

# parallelisation
import multiprocessing as mp

# astronomy
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
# from astropy.cosmology import CosmologyError   # this Error was weirdly not found!!

import SiGMo


# ================
# Helper Functions

def exclude_keys(in_dict, excl_keys) -> dict:
    """Return shallow(?) copy of dictionary in_dict without the key-value pairs designated by excl_keys"""
    return {key: in_dict[key] for key in in_dict if key not in excl_keys}

def make_copy_exclude_items_by_class(in_dict, excl_class) -> dict:
    """
    Excludes objects of a certain class/set of classes in a dictionary, and replaces them with their name attribute
    :param in_dict: dictionary from which the objects of specified class(es) shall be removed
    :param excl_class: class or list of classes, objects of which shall be removed (up to 1 list level deep)
    :return: returns a copy of the in_dict, with objects of type(s) excl_class removed and replaced by their name attr.
    """
    _tmp_exclude = {}
    for attr, value in in_dict.items():
        if isinstance(value, excl_class):  # straight-up instance of an AstroObject subclass
            _tmp_exclude[attr] = value.name
        elif isinstance(value, list):  # list of (possible) instances of AstroObject subclass(es)
            _tmp_exclude[attr] = []
            for value_i in value:
                if isinstance(value_i, excl_class):
                    _tmp_exclude[attr].append(value_i.name)
                else:
                    _tmp_exclude[attr].append(value_i)

    # copy the _tmp_in dict but exclude keys that have lists or straight-up AstroObjects as values
    _tmp_out = copy.deepcopy(exclude_keys(in_dict, _tmp_exclude.keys()))

    # update _tmp_out with the key-value pairs prev. excluded from deepcopy (and modified to remove AstroObjects)
    _tmp_out = _tmp_out | _tmp_exclude

    return _tmp_out



# ========================================================================
# Abstract Base Class for all AstroObjects (for Environment, Halo, Galaxy)

class AstroObject(ABC):
    """
    Abstract base class that just provides common attributes and methods for all different AstroObject subclasses

    Methods
    -------
    evolve()
        Abstract method that forces subclasses to implement their own 'evolve()' method
    make_snapshot(single_snapshot: bool = True)
        Returns the current values of all major attributes of this AstroObject as dict
    make_single_snapshot()
        Returns the current values of all major attributes as dict,
        replaces instances of the subclasses of AstroObject by their name attribute.
    make_multi_snapshot()
        Returns the current values of all major attributes as dict, including lower-in-hierarchy objects, and
        replaces instances of the higher-in-hierarchy subclasses of AstroObject by their name attribute
    make_and_write_all_snapshots(index: int, n_steps: int, outdir: Path, single_snapshots: bool = True)
        Writes snapshots of the Environment self and of all linked Halos and Galaxies to disk
    """

    @abstractmethod
    def __init__(self):
        self.name = None
        return

    @abstractmethod
    def evolve(self):
        """
        Abstract method that forces subclasses to implement their own 'evolve()' method
        :return: None
        """
        return

    def make_snapshot(self, single_snapshot: bool = True) -> 'Snapshot':
        """
        Returns the current values of all major attributes of this AstroObject as dict,
        and if single_snapshot=False, it also includes the cleaned-up versions of the lower-in-hierarchy
        AstroObjects in the Snapshot
        :param single_snapshot: (bool) If True, return flat-ish Snapshot that has replaced all linked AstroObjects
        with their name attribute; if False, return multi-snapshot that has replaced all above-in-hierarchy AstroObjects
        with their names, but includes all down-in-hierarchy AstroObjects as dicts (as if in owns flat-ish Snapshots).
        :return: Snapshot of the current object
        """
        if single_snapshot:
            return self.make_single_snapshot()
        else:
            return self.make_multi_snapshot()

    def make_single_snapshot(self) -> 'Snapshot':
        """
        Returns the current values of all major attributes as dict,
        replaces instances of the subclasses of AstroObject by their name attribute.
        (This is the original, already copy-optimised make_snapshot method that only ever did single snapshots)
        :return: flat-ish Snapshot of current object
        """
        _tmp_in = dict(vars(self))

        # search for lists and straight-up AstroObjects and build _tmp_exclude dict to exclude them from deepcopy.
        # replace instances and lists (with possible instances) of AstroObject subclasses with their 'name' attribute
        # use-case: env, halos, galaxies will usually contain those, but including them in the deepcopy results in
        # circular references and very deep copying (e.g. env contains galaxy, that has its env, that has the galaxy...)
        _tmp_exclude = {}
        for attr, value in _tmp_in.items():
            if isinstance(value, AstroObject):  # straight-up instance of an AstroObject subclass
                _tmp_exclude[attr] = value.name
            elif isinstance(value, list):  # list of (possible) instances of AstroObject subclass(es)
                _tmp_exclude[attr] = []
                for value_i in value:
                    if isinstance(value_i, AstroObject):
                        _tmp_exclude[attr].append(value_i.name)
                    else:
                        _tmp_exclude[attr].append(value_i)

        # copy the _tmp_in dict but exclude keys that have lists or straight-up AstroObjects as values
        _tmp_out = copy.deepcopy(exclude_keys(_tmp_in, _tmp_exclude.keys()))

        # update _tmp_out with the key-value pairs prev. excluded from deepcopy (and modified to remove AstroObjects)
        _tmp_out = _tmp_out | _tmp_exclude

        return SiGMo.Snapshot(_tmp_out)

    def make_multi_snapshot(self) -> 'Snapshot':
        """
        Returns the current values of all major attributes as dict, including lower-in-hierarchy objects, and
        replaces instances of the higher-in-hierarchy subclasses of AstroObject by their name attribute
        :return: deep-ish Snapshot of current object
        """
        _tmp_in = dict(vars(self))

        # search for lists and straight-up AstroObjects higher in the hierarchy and build _tmp_exclude dict to exclude
        # them from deepcopy. Replace higher-up instances and lists (with possible higher-up instances) of AstroObject
        # subclasses with their 'name' attribute.
        # use-case: env, halos, galaxies will usually contain those, but including them in the deepcopy results in
        # circular references and very deep copying (e.g. env contains galaxy, that has its env, that has the galaxy...)

        if isinstance(self, Environment):  # for the case of an Environment object: go 2 additional levels deep
            # do single, zero depth dict first (like in make_single_snapshot)
            _tmp_out = make_copy_exclude_items_by_class(in_dict=_tmp_in, excl_class=AstroObject)
            for i, halo in enumerate(self.halos):
                # replace the list of halo names by dicts with zero depth halo properties
                _tmp_halo_in = dict(vars(halo))
                _tmp_halo_out = make_copy_exclude_items_by_class(in_dict=_tmp_halo_in, excl_class=AstroObject)
                _tmp_out["halos"][i] = _tmp_halo_out
                for j, gal in enumerate(halo.galaxies):
                    # replace the list of galaxy names by dicts with zero depth galaxy properties
                    _tmp_gal_in = dict(vars(gal))
                    _tmp_gal_out = make_copy_exclude_items_by_class(in_dict=_tmp_gal_in, excl_class=AstroObject)
                    _tmp_out["halos"][i]["galaxies"][j] = _tmp_gal_out
            # generate snapshot from this dict
            _snp = SiGMo.Snapshot(_tmp_out)
        elif isinstance(self, Halo):  # for the case of a Halo object: go 1 additional level deep
            _tmp_out = make_copy_exclude_items_by_class(in_dict=_tmp_in, excl_class=AstroObject)
            for i, gal in enumerate(self.galaxies):
                # replace the list of galaxy names by dicts with zero depth galaxy properties
                _tmp_gal_in = dict(vars(gal))
                _tmp_gal_out = make_copy_exclude_items_by_class(in_dict=_tmp_gal_in, excl_class=AstroObject)
                _tmp_out["galaxies"][i] = _tmp_gal_out
            # generate snapshot from this dict
            _snp = SiGMo.Snapshot(_tmp_out)
        elif isinstance(self, Galaxy):  # for the case of a Galaxy object: no additional levels needed
            _snp = self.make_single_snapshot()
        else:
            raise TypeError(f'Objects of type {type(self)} are not supported by make_multi_snapshot()')

        return _snp

    def make_and_write_all_snapshots(self, index: int, n_steps: int, outdir: Path, single_snapshots: bool = True):
        """
        Writes snapshots of the Environment self and of all linked Halos and Galaxies to disk
        :param index: unique identifier for this whole set of snapshots, e.g. number of the timestep
        :param n_steps: maximum number of the identifier index
        :param outdir: output directory for all snapshot files
        :param single_snapshots: will single Snapshots and output files be made for this very AstroObject and every
        single AstroObject below it in hierarchy (True), or will one large, all-encompassing multi-Snapshot and file be
        made for this very AstroObject and its lower-in-hierarchy AstroObjects (False)?
        :return: None
        """
        # if we want to write single snapshot files to disk for every single object
        if single_snapshots:
            snp = self.make_snapshot(single_snapshot=True)
            snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
            # split depending on whether self is Environment, Halo, or Galaxy
            # Environment
            if isinstance(self, Environment):
                for halo in self.halos:
                    # write all Halos in the Environment to disk
                    snp = halo.make_snapshot(single_snapshot=True)
                    snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
                    for gal in halo.galaxies:
                        # write all Galaxies in this Halo to disk
                        snp = gal.make_snapshot(single_snapshot=True)
                        snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
            # Halo
            elif isinstance(self, Halo):
                for gal in self.galaxies:
                    # write all Galaxies in this Halo to disk
                    snp = gal.make_snapshot(single_snapshot=True)
                    snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
            # Galaxy
            elif isinstance(self, Galaxy):
                pass  # this snapshot was already done initially, ahead of the differentiation
            else:
                raise TypeError(f'Objects of type {type(self)} are not supported by make_and_write_all_snapshots()')
        # if we want to write one big snapshot file to disk for all the objects together
        else:
            snp = self.make_snapshot(single_snapshot=False)
            snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
        return


    def __repr__(self) -> str:
        """Return representation of respective AstroObject. Uses self.make_snapshot() to get values, which have
        therefore already been cleansed of any AstroObjects referenced in the attributes (replaced by their 'name')"""
        snapshot = self.make_snapshot(single_snapshot=True)
        r_string = ", ".join("=".join((str(k), repr(v))) for k, v in snapshot.data.items())

        return f'{type(self).__name__}({r_string})'


    def __str__(self) -> str:
        """Return more user-friendly output for AstroObject. Uses self.make_snapshot() to get values, which have
        therefore already been cleansed of any AstroObjects referenced in the attributes (replaced by their 'name')"""
        snapshot = self.make_snapshot(single_snapshot=True)
        s_string = "\n".join(" = ".join(("  " + str(k), str(v))) for k, v in snapshot.data.items())

        return f'Instance of {type(self).__name__}() with the following attributes:\n' + s_string


# ===============================
# Class for the environment that one or many halos and galaxies live(s) in

class Environment(AstroObject):
    """Class for simple environment of one or more halos of the Halo class.

    Attributes
    ----------
    age : float, optional
        The current age of the system in Gyrs (default 0.)
    galaxies : list, optional
        List of all Galaxy objects in this environment (default None)
    halos : list, optional
        List of all Halo objects in this environment (default None)
    lookbacktime : float, optional
        The current cosmic lookback time (default None)
    mdm : float, optional
        The amount of dark matter available in the environment (default np.inf)
    mgas : float, optional
        The amount of gas available in the environment (default np.inf)
    name : str, optional
        The name of the galaxy (default 'Test_Env')
    [previous : Snapshot, optional
        A snapshot of the previous state (default None)] set automatically
    [z : float, optional
        The current redshift of the system (default None)] set automatically
    zstart : float, optional
        The initial redshift of the system (default None)

    Methods
    -------
    create_Halo(*halo_args, **halo_kwargs)
        Creates Halo object and adds it to halos list
    evolve(mode: str = "intuitive", timestep: float = 1.e-3, runparallel: bool = False)
        Evolve Environment and all halos/galaxies either intuitively or per Lilly+13, Eq.12a-14a, acc. to 'mode'
    """

    def __init__(self,
                 age: float = 0.,
                 galaxies: list = None,
                 halos: list = None,
                 lookbacktime: float = None,
                 mdm: float = np.inf,
                 mgas: float = np.inf,
                 name: str = 'Test_Env',
                 zstart: float = None
                 ):
        self.age = age
        self.galaxies = galaxies if galaxies else []
        self.halos = halos if halos else []
        self.lookbacktime = lookbacktime
        self.mdm = mdm
        self.mgas = mgas
        self.name = name
        self.previous = None
        self.zstart = zstart

        # make either lookbacktime from zstart or vice versa
        # zstart --> lookbacktime
        if (lookbacktime is None) and (zstart is not None):
            self.lookbacktime = cosmo.lookback_time(zstart).value  # IMPLICIT HARDCODING OF Gyr from astropy routine!
        # lookbacktime --> zstart
        elif (zstart is None) and (lookbacktime is not None):
            self.zstart = z_at_value(cosmo.age, cosmo.age(0) - lookbacktime * u.Gyr)  # UNIT Gyr HARDCODED!!
        else:
            # neither supplied --> default zstart=6
            if (zstart is None) and (lookbacktime is None):
                print(f"Neither intial redshift nor lookback time are supplied!\n" +
                      f"Reverting back to default value of z=6 and corresponding\n" +
                      f"lookback time calculated from it.")
                self.zstart = 6.  # EXPLICIT HARD-CODING OF DEFAULT z=6
            # both supplied --> use zstart --> lookbacktime
            elif (zstart is not None) and (lookbacktime is not None):
                print(f"Both intial redshift and lookback time are supplied!\n" +
                      f"Will use the supplied value of z={zstart:9.2f} to calculate\n" +
                      f"the corresponding lookback time from it.")
            self.lookbacktime = cosmo.lookback_time(self.zstart).value  # IMPLICIT HARDCODING OF Gyr from astropy routine!

        # set self.z from self.zstart
        self.z = self.zstart
        return


    def create_Halo(self,
                    *halo_args,
                    **halo_kwargs
                    ):
        """Creates Halo object and adds it to halos list"""
        halo = Halo(env=self, *halo_args, **halo_kwargs)
        self.halos.append(halo)
        return halo


    def evolve(self,
               mode: str = "intuitive",
               timestep: float = 1.e-3,
               runparallel: bool = False
               ):
        """Evolve Environment and all halos/galaxies either intuitively or per Lilly+13, Eq.12a-14a, acc. to 'mode'"""
        # store snapshot of previous state before anything gets changed

        # # THIS IS ONLY NEEDED FOR THE AFTERWARD CALCULATION OF FRACTIONS (similar to Lilly) – commented out
        # self.previous = None
        # self.previous = self.make_snapshot()

        # make the time step in lookbacktime, then convert to z as well
        self.lookbacktime -= timestep
        self.age += timestep
        # self.z = z_at_value(cosmo.age, cosmo.age(0) - self.lookbacktime * u.Gyr)  # Gyr HARDCODED AGAIN!
        try:
            self.z = z_at_value(cosmo.age, cosmo.age(0) - self.lookbacktime * u.Gyr)  # Gyr HARDCODED AGAIN!
        except:
            z_at_value_sig = inspect.signature(z_at_value)
            z_at_value_params = z_at_value_sig.parameters
            zmin = z_at_value_params["zmin"].default
            if self.lookbacktime * u.Gyr <= cosmo.age(0) - cosmo.age(zmin):    # Gyr HARDCODED AGAIN!
                # use linear interpolation between z=zmin and z=0 instead of jumping straight to z=0 below zmin
                if self.lookbacktime > 0:
                    lin_fact = self.lookbacktime / cosmo.age(zmin).value
                    self.z = lin_fact * zmin
                else:
                    self.z = 0.
                    print("WARNING: at least one timestep overshot beyond redshift 0 !!!")
            else:
                raise

        # go through all the halos and evolve/update them based on new time
        if (mode == "reference") or (mode == "intuitive"):  # HARDCODED the two different modes
            if runparallel:
                with mp.Pool(mp.cpu_count()-1) as pool:    # HARDCODED the number of CPUs as available CPUs - 1
                    result_snps = pool.starmap(Halo.evolve, [(halo, mode, timestep) for halo in self.halos])
                    pool.close()
            else:
                for halo in self.halos:
                    halo.evolve(mode=mode, timestep=timestep)
        else:
            warnings.warn("Unsupported keyword for 'mode' in Environment.evolve(). \n"
                          ""f"Environment '{self.name}' was not evolved")

        return self.make_snapshot()

    # # Old, working method, previously just called write_all_snapshots. Now uses new method inherited from AstroObject
    # def make_and_write_all_snapshots(self, index, n_steps, outdir):
    #     """
    #     Writes snapshots of the Environment self and of all linked Halos and Galaxies to disk
    #     :param index: unique identifier for this whole set of snapshots, e.g. number of the timestep
    #     :param n_steps: maximum number of the identifier index
    #     :param outdir: output directory for all snapshot files
    #     :return: no return value
    #     """
    #     snp = self.make_snapshot()
    #     snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
    #     for halo in self.halos:
    #         # write all Halos in the Environment to disk
    #         snp = halo.make_snapshot()
    #         snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
    #         for gal in halo.galaxies:
    #             # write all Galaxies in this Halo to disk
    #             snp = gal.make_snapshot()
    #             snp.save_to_disk(outdir / snp.autoname_with_index(index, n_steps))
    #     return


# ========================================================
# Class for the halo that lives in one environment and that one or many galaxies live(s) in


class Halo(AstroObject):
    """Class for simple halo with one or more galaxies of the Galaxy class.

    Attributes
    ----------
    env : Environment
        The instance of Environment this instance of Halo lives in
    age : float, optional
        The current age of the system in Gyrs (default None)
    BDR : float, optional
        The ratio of (gaseous) baryonic to dark matter entering the halo (default 0.15)
    DCR : float, optional
        The dark matter mass change rate (accretion) of the halo (default None)
    [fgal : float, optional
        The fraction of baryons that enter the halo and make it all the way down
        into the "regulator system" to participate in star formation etc] moved to Galaxy class
    galaxies : list, optional
        List of all Galaxy objects in this environment (default None)
    GCR: float, optional
        The change rate ∂/∂t in gas mass content of the halo (default None)
    HLF : float, optional
        The halo loss fraction - fraction of baryons that get expelled by feedback
        not only from the galaxy but also from the halo altogether (default 0.1)
    lookbacktime : float, optional
        The current cosmic lookback time (default None)
    mdm : float, optional
        The amount of dark matter available in the halo (default None)
    mgas : float, optional
        The amount of gas available in the environment (default None)
    MIR : float, optional
        The total mass increase rate (accretion) of the halo with mass mtot (default None)
    mtot : float, optional
        The total mass of the halo that the galaxy resides in (default None)
    name : str, optional
        The name of the galaxy (default 'Test_Env')
    sMIR : float, optional
        The specific mass increase rate (accretion) of the DM halo (default None)
    sMIR_scaling : float, optional
        Artificial scaling (multiplicative increase or decrease) of the specific
        halo accretion rate, e.g. to explore increased accretion (default 1.)
    sMIR_scaling_basefactor : float, optional
        Base factor that is used by the function referenced in sMIR_scaling_updater (default 1.)
    sMIR_scaling_updater : function, optional
        Function used to update the sMIR_scaling at every time step in the evolution.
        If sMIR_scaling_updater = None, no updating will be performed.
        If sMIR_scaling_updater is a function, it will be called, with only the
        current instance of the halo as an argument, from which all other information
        like redshift etc. needs to be derived (default None)
    uRandDraw : float, optional
        constant drawn from a uniform random distribution, to be used e.g. for unique offsets (default None)
    z : float, optional
        The current redshift of the system (default None)
    [zstart : float, optional
        The initial redshift of the system (default 6.)] deprecated, now in Environment

    Methods
    -------
    create_Galaxy(with_burnin=False, with_burnin_dict=None, *galaxy_args, **galaxy_kwargs)
        Creates Galaxy object and adds it to galaxies list
    evolve(mode: str = "intuitive", timestep: float = 1.e-3)
        Evolve Halo and all galaxies either intuitively or per Lilly+13, Eq.12a-14a, acc. to 'mode'
    update_DCR(*args, **kwargs)
        Update the dark matter change rate using the compute_DCR() method
    compute_DCR(MIR: float  = None)
        Computes the dark matter change rate - should be made obsolete since MIR is that now
    update_GCR(*args, **kwargs)
        Update the gas mass change rate using the compute_GCR() method
    compute_GCR(BDR: float = None, gal_fgal: list[float] = None, gal_MLR: list[float] = None, HLF: float = None,
        MIR: float  = None)
        Computes the gas mass change rate
    update_mdm(timestep: float, *args, **kwargs)
        Update the dark matter mass using the compute_mdm() method
    compute_mdm(timestep: float, DCR: float = None, mdm: float = None)
        Compute the dark matter mass
    update_mgas(timestep: float, *args, **kwargs)
        Update the gas mass using the compute_mgas() method
    compute_mgas(timestep: float, gal_MLR: list[float] = None, GCR: float = None, mgas: float = None)
        Compute the gas mass
    update_mtot(*args, **kwargs)
        Update the total halo mass using the compute_mtot() method
    compute_mtot(mdm: float = None, mgas: float = None, gal_mgas: list[float] = None, gal_mstar: list[float] = None)
        Compute new total halo mass based on the halo's DM and gas reservoirs
        (mdm and mgas) as well as all contained galaxies' gas reservoirs (mgas)
        and stellar masses (mstar). Those are time-integrated quantities
        This ins NOT based on previous values of mtot, as this is just a summary value.
    update_MIR(sMIR: float = None, *args, **kwargs)
        Update the current dark matter mass increase rate
    compute_MIR(mtot: float = None, sMIR: float = None)
        Computes the current DARK MATTER Mass Increase Rate from sMIR
    update_sMIR(sMIR_scaling_updater = None, *args, **kwargs)
        Update the specific mass increase rate (of dark matter)
    compute_sMIR(mtot: float = None, sMIR_scaling: float = None, z: float = None)
        Computes the specific Mass Increase Rate of the DM halo
        accoding to Lilly et al. 2013, Eq. (3), more precise version.
        Modified to include an optional scaling factor to regulate
        accretion rate on the halo in total, as well as an optional
        update to this scaling factor
    update_sMIR_scaling(*args, **kwargs)
        Update the sMIR_scaling
    compute_sMIR_scaling(updater_function)
        Computes the updated sMIR_scaling using the function handed in
        and the current environment instance
    """

    def __init__(self,
                 env: Environment,
                 age: float = None,
                 BDR: float = 0.15,
                 DCR: float = None,  # used to be 0.
                 # fgal: float = 1.,
                 galaxies: list = None,
                 GCR: float = None,  # used to be 0.
                 HLF: float = 0.1,
                 lookbacktime: float = None,
                 mdm: float = None,
                 mgas: float = None,
                 MIR: float = None,
                 mtot: float = None,
                 name: str = "Test_Halo",
                 sMIR: float = None,
                 sMIR_scaling = 1.,
                 sMIR_scaling_basefactor = 1.,
                 sMIR_scaling_updater = None,
                 uRandDraw = None,
                 z: float = None
                 ):
        self.env = env
        self.age = self.env.age if age is None else age
        self.BDR = BDR
        self.DCR = DCR
        # self.fgal = fgal
        self.galaxies = [] if galaxies is None else galaxies
        self.GCR = GCR
        self.HLF = HLF
        self.lookbacktime = self.env.lookbacktime if lookbacktime is None else lookbacktime
        self.mdm = mdm
        self.mgas = mgas
        self.MIR = MIR
        self.mtot = self.compute_mtot() if mtot is None else mtot
        self.name = name
        self.previous = None
        self.sMIR = sMIR
        self.sMIR_scaling = sMIR_scaling
        self.sMIR_scaling_basefactor = sMIR_scaling_basefactor
        self.sMIR_scaling_updater = sMIR_scaling_updater
        if uRandDraw is None:
            rng = np.random.default_rng()
            self.uRandDraw = rng.uniform(low=0.0, high=1.0, size=None)
        else:
            self.uRandDraw = uRandDraw
        self.z = env.z if z is None else z

        # re-set sMIR and MIR: don't want to set them properly earlier b/c order of attr. wouldn't be alphabetic
        self.sMIR = self.compute_sMIR() if self.sMIR is None else sMIR
        self.MIR = self.compute_MIR() if self.MIR is None else self.MIR
        # re-set DCR and GCR: they depend on MIR, and GCR on some additional stuff
        self.DCR = self.compute_DCR() if self.DCR is None else self.DCR
        self.GCR = self.compute_GCR() if self.GCR is None else self.GCR
        return


    def create_Galaxy(self,
                      with_burnin=False,
                      with_burnin_dict=None,
                      *galaxy_args,
                      **galaxy_kwargs
                      ):
        """Creates Galaxy object and adds it to galaxies list"""
        if with_burnin:
            if not with_burnin_dict:
                gal = Galaxy.with_burnin(env=self.env, halo=self, **galaxy_kwargs)
            else:
                gal = Galaxy.with_burnin(**with_burnin_dict, env=self.env, halo=self, **galaxy_kwargs)
        else:
            gal = Galaxy(env=self.env, halo=self, **galaxy_kwargs)  # is direct reference to env still necessary?
        self.galaxies.append(gal)
        return gal


    def evolve(self,
               mode: str = "intuitive",
               timestep: float = 1.e-3
               ):
        """Evolve Halo and all galaxies either intuitively or per Lilly+13, Eq.12a-14a, acc. to 'mode'"""

        # # THIS IS ONLY NEEDED FOR THE AFTERWARD CALCULATION OF FRACTIONS (similar to Lilly) – commented out
        # # store snapshot of previous state before anything gets changed
        # self.previous = None
        # self.previous = self.make_snapshot()

        # take lookbacktime and z (redshift) from Environment the Halo belongs to
        self.lookbacktime = self.env.lookbacktime
        self.z = self.env.z
        self.age += timestep

        # update the time-variable quantities involved, in this case
        # MIR and through (and somewhat before) it sMIR
        self.update_MIR()

        # go through all the galaxies and evolve/update them
        if (mode == "reference") or (mode == "intuitive"):
            for galaxy in self.galaxies:
                galaxy.evolve(mode=mode, timestep=timestep)
        else:
            warnings.warn("Unsupported keyword for 'mode' in Halo.evolve(). \n"
                          ""f"Halo '{self.name}' was not evolved")

        # update the mass change rates for the halo's mass reservoirs (dark matter and gas)
        self.update_DCR()
        self.update_GCR()

        # update the mass reservoirs of the halo itself excluding galaxies (dark matter and gas)
        self.update_mdm(timestep=timestep)
        self.update_mgas(timestep=timestep)

        # update the (summary) total mass of the halo including all masses of the galaxies
        self.update_mtot()

        return self.make_snapshot()


    # ============================
    # computing physical quantities of the halo

    # DCR
    def update_DCR(self, *args, **kwargs):
        """Update the dark matter change rate using the compute_DCR() method"""
        self.DCR = self.compute_DCR(*args, **kwargs)
        return self.DCR

    def compute_DCR(self,
                    # BDR: float = None,
                    MIR: float  = None
                    ):
        """
        Computes the dark matter change rate - should be made obsolete since MIR is that now
        :param MIR: specific mass increase rate (of dark matter only, actually)
        :return: dark matter change rate - now the same as MIR
        """
        # BDR = self.BDR if BDR is None else BDR
        MIR = self.MIR if MIR is None else MIR

        # return MIR * (1. - BDR)   # old prescription assuming MIR was not only DM, but DM+gas, which it isn't!
        return MIR * 1.


    # GCR
    def update_GCR(self, *args, **kwargs):
        """Update the gas mass change rate using the compute_GCR() method"""
        self.GCR = self.compute_GCR(*args, **kwargs)
        return

    def compute_GCR(self,
                    BDR: float = None,
                    gal_fgal: list[float] = None,
                    gal_MLR: list[float] = None,
                    HLF: float = None,
                    MIR: float  = None
                    ):
        """
        Computes the gas mass change rate

        :param BDR: barion-dark matter ratio
        :param gal_fgal: list of fractions of gas entering the halo that reaches the star-forming region of each galaxy
        :param gal_MLR: list of mass loss rates of all individual galaxies in the halo
        :param HLF: halo loss fraction
        :param MIR: dark matter mass increase rate of the halo (not! specific MIR)
        :return: GCR
        """
        BDR = self.BDR if BDR is None else BDR
        HLF = self.HLF if HLF is None else HLF
        MIR = self.MIR if MIR is None else MIR

        # sum up all 'fractions' of how much gas reaches each galaxy
        gal_fgal_sum = 0
        if gal_fgal is None:
            for gal in self.galaxies:
                gal_fgal_sum += gal.fgal
        else:
            for gal_fgal_i in gal_fgal:
                gal_fgal_sum += gal_fgal_i
        if gal_fgal_sum > 1:
            print("WARNING: Sum of individual galaxies' fgal > 1!"
                  "         Galaxies accreting more than their Halo not yet supported in this model!")

        gal_MLR_sum = 0
        if gal_MLR is None:
            for gal in self.galaxies:
                gal_MLR_sum += gal.MLR
        else:
            for gal_MLR_i in gal_MLR:
                gal_MLR_sum += gal_MLR_i

        # return ((1. - gal_fgal_sum) * MIR * BDR) + ((1. - HLF) * gal_MLR_sum)   # Does MLR cut off fast enough when galaxy is depleted?
        # (above) That was the old prescription assuming MIR included not ony DM, but also gas, which it doesn't.
        return ((1. - gal_fgal_sum) * MIR * BDR) + ((1. - HLF) * gal_MLR_sum)   # Does MLR cut off fast enough when galaxy is depleted?


    # mdm
    def update_mdm(self, timestep: float, *args, **kwargs) -> float:
        """Update the dark matter mass using the compute_mdm() method"""
        self.mdm = self.compute_mdm(timestep, *args, **kwargs)
        return self.mdm

    def compute_mdm(self,
                    timestep: float,
                    DCR: float = None,
                    mdm: float = None
                    ) -> float:
        """
        Compute the dark matter mass

        :param timestep: timestep length
        :param DCR: dark matter mass change rate (same as MIR now)
        :param mdm: dark matter mass up until now
        :return: the new, larger dark matter mass
        """
        DCR = self.DCR if DCR is None else DCR
        mdm = self.mdm if mdm is None else mdm

        mdm += (DCR * timestep)
        return mdm if mdm > 0. else 0.


    # mgas
    def update_mgas(self, timestep: float, *args, **kwargs) -> float:
        """Update the gas mass using the compute_mgas() method"""
        self.mgas = self.compute_mgas(timestep, *args, **kwargs)
        return self.mgas

    def compute_mgas(self,
                     timestep: float,
                     gal_MLR: list[float] = None,
                     GCR: float = None,
                     mgas: float = None,
                     ) -> float:
        """
        Compute the gas mass

        :param timestep:
        :param gal_MLR:
        :param GCR:
        :param mgas:
        :return:
        """
        GCR = self.GCR if GCR is None else GCR
        mgas = self.mgas if mgas is None else mgas

        mgas += (GCR * timestep)

        return mgas if mgas > 0. else 0.


    # mtot (formerly mhalo)
    def update_mtot(self, *args, **kwargs) -> float:
        """Update the total halo mass using the compute_mtot() method"""
        self.mtot = self.compute_mtot(*args, **kwargs)
        return self.mtot

    def compute_mtot(self,
                     mdm: float = None,
                     mgas: float = None,
                     gal_mgas: list[float] = None,
                     gal_mstar: list[float] = None
                     ) -> float:
        """Compute new total halo mass based on the halo's DM and gas reservoirs
        (mdm and mgas) as well as all contained galaxies' gas reservoirs (mgas)
        and stellar masses (mstar). Those are time-integrated quantities
        This ins NOT based on previous values of mtot, as this is just a summary value."""
        mdm = self.mdm if mdm is None else mdm
        mgas = self.mgas if mgas is None else mgas

        # sum of all galaxies' mgas
        gal_mgas_sum = 0
        if gal_mgas is None:
            for gal in self.galaxies:
                gal_mgas_sum += gal.mgas
        else:
            for gal_mgas_i in gal_mgas:
                gal_mgas_sum += gal_mgas_i

        # sum of all galaxies' mstar
        gal_mstar_sum = 0
        if gal_mstar is None:
            for gal in self.galaxies:
                gal_mstar_sum += gal.mstar
        else:
            for gal_mstar_i in gal_mstar:
                gal_mstar_sum += gal_mstar_i

        return mdm + mgas + gal_mgas_sum + gal_mstar_sum


    # MIR
    def update_MIR(self, sMIR: float = None, *args, **kwargs) -> float:
        """Update the current dark matter mass increase rate"""
        if sMIR is None:
            self.update_sMIR()

        self.MIR = self.compute_MIR(*args, **kwargs)
        return self.MIR

    def compute_MIR(self,
                    mtot: float = None,
                    sMIR: float = None
                    ) -> float:
        """
        Computes the current DARK MATTER Mass Increase Rate from sMIR
        :param mtot: Total halo mass, including all masses of all galaxies in the halo
        :param sMIR: Specific DM mass increase rate, so DM mass increase per unit DM mass
        :return: MIR, the current DM mass increase rate
        """
        mtot = self.mtot if mtot is None else mtot
        sMIR = self.sMIR if sMIR is None else sMIR

        return sMIR * mtot


    # sMIR
    def update_sMIR(self, sMIR_scaling_updater = None, *args, **kwargs) -> float:
        """Update the specific mass increase rate (of dark matter)"""
        # if necessary: update the sMIR_scaling prior to applying it in compute_sMIR()
        sMIR_scaling_updater = self.sMIR_scaling_updater if sMIR_scaling_updater is None else sMIR_scaling_updater
        if sMIR_scaling_updater is not None:
            self.update_sMIR_scaling(sMIR_scaling_updater)

        self.sMIR = self.compute_sMIR(*args, **kwargs)
        return self.sMIR

    def compute_sMIR(self,
                     mtot: float = None,
                     sMIR_scaling: float = None,
                     z: float = None
                     ) -> float:
        """
        Computes the specific Mass Increase Rate of the DM halo
        accoding to Lilly et al. 2013, Eq. (3), more precise version.
        Modified to include an optional scaling factor to regulate
        accretion rate on the halo in total, as well as an optional
        update to this scaling factor

        :param mtot:
        :param sMIR_scaling:
        :param z:
        :return:
        """
        mtot = self.mtot if mtot is None else mtot
        sMIR_scaling = self.sMIR_scaling if sMIR_scaling is None else sMIR_scaling
        z = self.z if z is None else z

        return sMIR_scaling * 0.027 * (mtot / 10 ** 12) ** (0.15) * (1 + z + 0.1 * ((1 + z) ** (-1.25))) ** 2.5


    # sMIR_scaling
    def update_sMIR_scaling(self, *args, **kwargs) -> float:
        """Update the sMIR_scaling"""
        self.sMIR_scaling = self.compute_sMIR_scaling(*args, **kwargs)
        return self.sMIR_scaling

    def compute_sMIR_scaling(
            self,
            updater_function
    ) -> float:
        """
        Computes the updated sMIR_scaling using the function handed in
        and the current environment instance

        :param updater_function: function used to compute the new value of
            sMIR_scaling
        :return: the new value for sMIR_scaling
        """
        return updater_function(self)


# ========================================================
# Class for the galaxy that lives in one halo, which in turn lives in one environment


class Galaxy(AstroObject):
    """Class for simple model of a galaxy. Every galaxy should be associated
    with exactly one Halo object (and through it with exactly one Environment
    object), which provides a method to create a galaxy and add it to its galaxy list.

    Attributes
    ----------
    env : Environment
        The environment object the galaxy is associated with/its halo is located in
    halo : Halo
        The halo object the galaxy is associated with/is located in
    age : float, optional
        The current age of the system in Gyrs (default None)
    [BDR : float, optional
        The ratio of (gaseous) baryonic to dark matter entering the halo (default 0.2)] moved to Halo object
    fgal : float, optional
        The fraction of baryons that enter the halo and make it all the way down
        into the "regulator system" to participate in star formation etc (default 0.1)
    fgas: float, optional
        Gas mass fraction following Lilly+13 - not usually used anymore (default None)
    fout: float, optional
        Outflow mass fraction following Lilly+13 - not usually used anymore (default None)
    fstar: float, optional
        Stellar mass fraction following Lilly+13 - not usually used anymore (default None)
    GAR : float, optional
        The gas accretion rate of the galaxy (default 0.)
    GCR : float, optional
        The change rate ∂/∂t in gas mass content of the galaxy (default 0.)
    [HLF : float, optional
        The halo loss fraction - fraction of baryons that get expelled by feedback
        not only from the galaxy but also from the halo altogether (default 0.1)] moved to Halo object
    IRF: float, optional
        The fraction of gas being converted to stars that is promptly,
        here instantly, returned to the gas reservoir (default 0.4)
    lookbacktime: float, optional
        The current cosmic lookback time (default None)
    [MIR : float, optional
        The mass increase rate (accretion) of the DM halo with mass mhalo (default 0.)] moved to Halo object
    MLF : float, optional
        The mass-loading factor coupling SFR and mass loss (default 0.1)
    MLR : float, optional
        The mass loss rate from the galaxy (default 0.)
    macc : float, optional
        The total mass accreted onto the galaxy (default 0.)
    mgas : float, optional
        The gas mass content of the galaxy (default 1.e10)
    [mhalo : float, optional
        The total mass of the halo that the galaxy resides in (default 1.e12)] moved to Halo object
    mstar : float, optional
        The stellar mass content of the galaxy (default 1.e9)
    mout: float, optional
        The mass lost from the system due to massloss (default 0.)
    name : str, optional
        The name of the galaxy (default 'Test_Gal')
    [previous : dict, optional
        Stores the previous state of all galaxy properties to allow easy
        computation of delta_YYY change properties. Gets overwritten at
        the beginning of any XXX_evolve() method (default None)] gets set automatically later
    rsSFR : float, optional
        The reduced specific SFR; excludes instant.returned gas (default 0.)
    SFE : float, optional
        The star formation efficiency (default 1.)
    SFR : float, optional
        The star formation rate in the galaxy (default 0.)
    [sMIR : float, optional
        The specific mass increase rate (accretion) of the DM halo (default 0.)] moved to Halo object
    sSFR: float, optional
        The actual specific star formation rate; this sSFR does not account
        for reduction by the inst. return to the gas reservoir (default 0.)
    z: float, optional
        The current redshift of the galaxy, gets it from Environment (default None)

    Methods
    -------
    > methods for burn-in:
    with_burnin(cls, cycle_steps: int = 5, cycles_max: int = 1e6, check_attr: dict = None, div_aim: float = 1.e-3,
        div_delta_aim: float = 1.e-4, div_max: float = 1.e3, fixed_attr: dict = None, vartime: float = 1.e-3, *args,
        **kwargs)
        Allows for 'burn-in' of select galaxy properties and for others to be kept fixed. Aim: (quasi-)equilibrium
    attr_to_dict(attr = None, default_attr = None)
        Make dict from list or dict of desired or default attributes using object attribute values
    check_for_convstabdiv(attr_to_check: dict, div_aim = 1.e-3, div_delta_aim = None, div_max = None, div_previous = None)
        Check for convergence, stability, divergence in attributes that are specified
    reset_attributes(attr_to_reset: dict)
        Resets attributes from dict. Default resets mhalo,mgas,mout to value of prev. timestep, keeping them fixed

    > methods for time-integration:
    evolve(mode: str = "intuitive", timestep: float = 1.e-3)
        Evolves the galaxy by one 'timestep' according to different prescriptions specified in 'mode'
    intuitive_evolve(timestep: float = 1.e-3)
        Evolves the galaxy by one timestep intuitively, *without* using
        sSFR(mstar, z) as time-dependent input (the latter is what Lilly+13 do)
    reference_evolve(timestep: float = 1.e-3)
        Evolves the galaxy by one timestep according to Lilly+13 Eq.12a,13a,14a,
        ideal regulator

    > methods for updating physical quantities
    update_GAR(MIR: float = None, *args, **kwargs)
        Updates the (absolute) gas accretion rate of the galaxy
        and calls compute_GAR() to get the new value. If the overall
        halo mass increase rate (MIR) is not provided, it also calls
        update_MIR() to compute the current value of that.
    compute_GAR(MIR: float = None, BDR: float = None, fgal: float = None)
        Compute the absolute gas accretion rate of the galaxy
        from the overall halo mass increase rate (MIR) as well as
        the baryon-dark matter ratio (BDR) and the fraction of
        infalling gas that makes it deep enough into the galaxy
        to participate in star formation etc.
    update_GCR(mode:str = "from fgas", *args, **kwargs)
        Update the gas change rate of the reservoir either intuitively, or based on fractions like Lilly+13
    compute_GCR_from_fgas(GAR: float = None, fgas: float = None)
        Compute (reservoir) gas change rate from fractions in Lilly+13 ideal regulator
    compute_GCR_from_GAR(GAR: float = None, IFR: float = None, MLR: float = None, SFR: float = None)
        Compute (reservoir) gas change rate more 'intuitively' straight
        from overall accretion of gas into the galaxy, minus the gas that is
        expelled or turned into (long or short lived aka any kind of) stars.
        [The instant return of gas from short lived stars is handled at the
        level of the integration of mstar, mgas] this has been changed
    update_macc(timestep: float, *args, **kwargs)
        Update the total mass accreted onto the galaxy
    compute_macc(timestep: float, GAR: float = None, macc: float = None)
        Compute the new total mass accreted by the galaxy
    update_mgas(timestep: float, *args, **kwargs)
        Update the gas mass (in reservoir)
    compute_mgas(timestep: float, GCR: float = None, mgas: float = None)
        Compute the new gas mass (in reservoir) from GCR that already factors in a SFR reduced by (1 - IRF)
    update_MLR(mode:str = "from fout", *args, **kwargs)
        Update the mass loss rate, which is the rate of mass that gets ejected from the galaxy
    compute_MLR_from_fout(GAR: float = None, fout: float = None)
        Compute mass loss rate from fractions in Lilly+13 ideal regulator
    compute_MLR_from_MLF(MLF: float = None, SFR: float = None)
        Compute mass loss rate 'intuitively' from mass-loading factor
        and star formation rate
    update_mout(timestep: float, *args, **kwargs)
        Update the total mass gas lost from the system up to this point
    compute_mout(timestep: float, MLR: float = None, mout: float = None)
        Compute the total gas mass lost from the system until now
    update_mstar(timestep: float, mode: str = "from reduced SFR", *args, **kwargs)
        Update current stellar mass according to the 'mode' keyword provided
    compute_mstar_from_reduced_SFR(timestep: float, mstar: float = None, SFR: float = None)
        Compute current stellar mass from SFR already reduced by (1 - IRF)
    compute_mstar_from_unreduced_SFR(timestep: float, IRF: float = None, mstar: float = None, SFR: float = None)
        Compute current stellar mass from SFR unreduced by (1 - IRF)
    update_rsSFR(mode: str = 'empirical', *args, **kwargs)
        Update the reduced specific star-formation rate
    compute_rsSFR_empirical(mstar: float = None, z: float = None)
        Computes the reduced specific star formation rate according to
        Lilly et al. 2013, Eq. (2), for z < 2. and z > 2., respectively
    compute_rsSFR_from_sSFR(IRF: float = None, sSFR:float = None)
        Computes the reduced specific star formation rate from
        the non-reduced, regular specific star formation rate
    update_SFR(mode: str = 'from fstar', *args, **kwargs)
        Compute and update SFR according to 'mode' keyword
    compute_SFR_from_fstar(GAR: float = None, fstar: float = None)
        Compute star formation rate from fractions in Lilly+13 ideal regulator
    compute_SFR_from_SFE(mgas: float = None, SFE: float = None)
        Compute star formation rate intuitively from star formation efficiency
        and available gas
    update_sSFR(mode: str = 'from rsSFR', rsSFR: float = None, *args, **kwargs)
        Updates the actual, unreduced sSFR and, if not supplied, the (reduced) rsSFR too
    compute_sSFR_from_rsSFR(rsSFR: float = None, IRF: float = None)
        Computes the actual, unreduced sSFR from the rsSFR
    compute_sSFR_from_SFR(mstar: float = None, SFR: float = None)
        Computes the intuitive sSFR from SFR and mstar

    > methods for updating fractions
    update_fgas(mode: str = "beforehand", *args, **kwargs)
        Update the fraction of gas accreted onto the galaxy that goes into the reservoir
    compute_fgas_afterwards(delta_macc: float = None, delta_mgas: float = None)
        Fraction of incoming gas added to the galaxy gas reservoir,
        based on the mgas and macc increase
    compute_fgas_beforehand(IRF: float = None, MLF: float = None, SFE: float = None, sSFR: float = None)
        Fraction of incoming gas added to the galaxy gas reservoir,
        based on Eq. 14a from Lilly+13, for ideal regulator (fres)
    update_fstar(mode: str = "beforehand", *args, **kwargs)
        Update the fraction of gas accreted onto the galaxy turned into long-lived stars
    compute_fstar_afterwards(delta_macc: float = None, delta_mstar: float = None)
        Fraction of incoming gas converted to (long lived) stars,
        based on the mstar and macc increase
    compute_fstar_beforehand(IRF: float = None, MLF: float = None, SFE: float = None, sSFR: float = None)
        Fraction of incoming gas converted to stars,
        based on Eq. 12a from Lilly+13, for ideal regulator
    update_fout(mode: str = "beforehand", *args, **kwargs)
        Update the fraction of gas accreted onto the galaxy expelled out of the galaxy again
    compute_fout_afterwards(delta_macc: float = None, delta_mout: float = None)
        Fraction of incoming gas expelled again from the galaxy,
        based on the mout and macc increase
    compute_fout_beforehand(IRF: float = None, MLF: float = None, SFE: float = None, sSFR: float = None)
        Fraction of incoming gas expelled again from the galaxy,
        based on Eq. 13a from Lilly+13, for ideal regulator
    """

    def __init__(self,
                 env: Environment,
                 halo: Halo,
                 age: float = None,
                 # BDR: float = 0.2,
                 fgal: float = 0.1,
                 fgas: float = None,
                 fout: float = None,
                 fstar: float = None,
                 GAR: float = 0.,  # THIS NEEDS TO BE CHANGED TO None AND THEN PROPERLY SET FURTHER DOWN
                 # gasmassfraction: float = 1.,
                 GCR: float = 0.,  # THIS NEEDS TO BE CHANGED TO None AND THEN PROPERLY SET FURTHER DOWN
                 # HLF: float = 0.1,
                 IRF: float = 0.4,
                 lookbacktime: float = None,
                 # MIR: float = 0.,
                 MLF: float = 0.1,
                 MLR: float = 0.,  # THIS NEEDS TO BE CHANGED TO None AND THEN PROPERLY SET FURTHER DOWN
                 macc: float = 0.,
                 mgas: float = 1.e10,
                 # mhalo: float = 1.e12,
                 mstar: float = 1.e9,
                 mout: float = 0.,
                 name: str = 'Test_Gal',
                 rsSFR: float = 0.,  # THIS NEEDS TO BE CHANGED TO None AND THEN PROPERLY SET FURTHER DOWN
                 SFE: float = 1.,
                 SFR: float = 0.,  # THIS NEEDS TO BE CHANGED TO None AND THEN PROPERLY SET FURTHER DOWN
                 # sMIR: float = 0.,
                 sSFR: float = 0.,  # THIS NEEDS TO BE CHANGED TO None AND THEN PROPERLY SET FURTHER DOWN
                 z: float = None
                 ):
        self.env = env
        self.halo = halo
        self.age = self.halo.age if age is None else age
        # self.BDR = BDR
        self.fgal = fgal
        self.fgas = fgas
        self.fout = fout
        self.fstar =fstar
        self.GAR = GAR
        self.GCR = GCR
        # self.HLF = HLF
        self.IRF = IRF
        self.lookbacktime = self.halo.lookbacktime if lookbacktime is None else lookbacktime
        # self.MIR = MIR
        self.MLF = MLF
        self.MLR = MLR
        self.macc = macc
        self.mgas = mgas
        # self.mhalo = mhalo
        self.mstar = mstar
        self.mout = mout
        self.name = name
        self.previous = None
        self.rsSFR = rsSFR
        self.SFE = SFE
        self.SFR = SFR
        # self.sMIR = sMIR
        self.sSFR = sSFR
        self.z = self.halo.z if z is None else z
        return


    # -------------------
    # methods for burn-in

    # classmethod to create a galaxy and 'burn it in' by trying to find a (quasi-)equilibrium state
    @classmethod
    def with_burnin(cls,
                    cycle_steps: int = 5,
                    cycles_max: int = 1e6,
                    check_attr: dict = None,
                    div_aim: float = 1.e-3,
                    div_delta_aim: float = 1.e-4,
                    div_max: float = 1.e3,
                    fixed_attr: dict = None,
                    vartime: float = 1.e-3,
                    *args,
                    **kwargs
                    ) -> 'Galaxy':
        """Allows for 'burn-in' of select galaxy properties and for others to be kept fixed. Aim: (quasi-)equilibrium"""
        # create new galaxy with initial values
        newgal = cls(*args, **kwargs)

        # set dict with attributes to evaluate to default, or supplement it with object attribute values, if necessary
        default_check_attr = ['mstar', 'SFR', 'GCR', 'MLR']
        check_attr = newgal.attr_to_dict(check_attr, default_check_attr)

        # set dict with fixed attributes to default, or supplement it with object attribute values, if necessary
        default_fixed_attr = ['mhalo', 'mgas']
        fixed_attr = newgal.attr_to_dict(fixed_attr, default_fixed_attr)

        # make sure z and mout stay fixed during burn-in no matter the user-input
        fixed_attr.update({'z': newgal.z, 'mout': newgal.mout})

        # set div to highest allowed value as initial value
        div = div_max

        # set initial values for some of the loop criteria
        converged = False
        cycle = 0

        # loop to make max cycles_max steps of burn-in
        while not converged:
            # store the attributes to be checked before next timestep
            for attr, value in check_attr.items():
                check_attr[attr] = getattr(newgal, attr)

            # evolve for specified nuber of timesteps within this one cycle
            for i in range(cycle_steps):
                newgal.intuitive_evolve(timestep=vartime)

            # check for convergence, stability and divergence
            div, attr_div, converged, stable, diverged = newgal.check_for_convstabdiv(attr_to_check=check_attr,
                                                                                      div_aim=div_aim,
                                                                                      div_delta_aim=div_delta_aim,
                                                                                      div_max=div_max)
            print("div =", div)

            attr_div_str = "\n                 ".join(f"{key} = {value}" for key, value in attr_div.items())

            if converged:
                print(f"Burn-in converged after {cycle} cycles ({cycle_steps} timesteps each)")
                print(f"Overall div:     div = {div} < {div_aim} = div_aim")
                print(f"Individual divs: " + attr_div_str)
                break
            else:
                if diverged:
                    print(f"Warning: burn-in diverged after {cycle} cycles ({cycle_steps} timesteps each)!")
                    break
                else:
                    newgal.reset_attributes(fixed_attr)
            cycle += 1
            if cycle == cycles_max:
                print(f"Warning: burn-in has not converged after maximum of {cycle} cycles ({cycle_steps} timesteps each),")
                if stable:
                    print(f"but the change in attributes checked is stable")
                else:
                    print(f"and the change in attributes checked is not stable")
                break

        return newgal


    # helper-method to set object attribute values as dict values
    def attr_to_dict(self,
                     attr = None,
                     default_attr = None
                     ) -> dict:
        """Make dict from list or dict of desired or default attributes using object attribute values"""

        # if attr dict or list is not supplied: use default or abort
        if attr is None:
            # if default_attr are supplied
            if default_attr is not None:
                attr = default_attr
            else:
                print("Setting values from object attributes failed:\n"
                      "Neither input nor default attributes set")

        # if attr was already supplied as dict, but values not set (set to None)
        if type(attr) is dict:
            for key, value in attr.items():
                if value is None:
                    attr[key] = getattr(self, key)
        # if attr was supplied as list
        elif type(attr) is list:
            new_check_attr = {}
            for key in attr:
                new_check_attr[key] = getattr(self, key)
            attr = new_check_attr

        return attr


    # check for convergence, stability and divergence in specified attributes
    def check_for_convstabdiv(self,
                              attr_to_check: dict,
                              div_aim = 1.e-3,
                              div_delta_aim = None,
                              div_max = None,
                              div_previous = None
                              ) -> (float, bool, bool):
        """Check for convergence, stability, divergence in attributes that are specified

        In this context,
          * converged: div of current iteration has fallen below a defined threshold (div_aim), so the change in
                       attribute is sufficiently small;
          * stable:    the change in div compared to the last iteration is sufficiently small (div_delta_aim). This
                       does not mean that the attribute does not change anymore, but just that the change is constant
                       albeit over one timestep only;
          * diverged:  div of current iteration has surpassed a defined threshold value (div_max), so the change is
                       deemed to be too extreme

        Returns: div (a measure for the change in all checked parameters),
                 converged (bool),
                 stable (bool),
                 diverged (bool)"""

        div_delta_aim = (div_aim / 10.) if div_delta_aim is None else div_delta_aim
        div_max = (div_aim * 1.e6) if div_max is None else div_max
        div_previous = div_max if div_previous is None else div_previous

        # get and compute individual attributes' divs
        attr_div = {}
        for attr, value_old in attr_to_check.items():
            value_new = getattr(self, attr)
            attr_div[attr] = (value_new - value_old) / (value_new + value_old + 1.e-7)  # avoid division by zero

        print("attr_div =", attr_div)

        # compute combined div
        div = np.array(list(attr_div.items()))[..., 1]
        div = div.astype(float)
        div = np.sqrt(np.sum(div**2))

        # calculate the change in div compared to previous iteration
        div_delta = abs(div - div_previous)

        # determine if converged, stable, diverged
        converged = True if div <= div_aim else False
        stable = True if div_delta <= div_delta_aim else False
        diverged = True if div > div_max else False

        return div, attr_div, converged, stable, diverged


    # reset attributes that are specified (e.g. that are supposed to be fixed)
    def reset_attributes(self,
                         attr_to_reset: dict
                         ) -> 'Galaxy':
        """Resets attributes from dict. Default resets mhalo,mgas,mout to value of prev. timestep, keeping them fixed"""

        # reset each attribute in turn
        for attr, value in attr_to_reset.items():
            setattr(self, attr, value)

        return self


    # ------------------------------
    # time integration of the system - reference method by Lilly+13, "ideal regulator"

    # This is the reference model from Lilly et al. 2013, ideal case
    # based on Eqns. (12a), (13a), (14a)
    # fractional splitting of incoming baryons, cf.Lilly+13,Eqns. 12-14a



    def reference_evolve(self,
                         timestep: float = 1.e-3
                         ) -> 'Snapshot':
        """Evolves the galaxy by one timestep according to Lilly+13 Eq.12a,13a,14a,
        ideal regulator"""

        # # THIS IS ONLY NEEDED FOR THE AFTERWARD CALCULATION OF FRACTIONS (similar to Lilly) – commented out
        # # store snapshot of previous state before anything gets changed
        # self.previous = None
        # self.previous = self.make_snapshot()

        # update the lookbacktime and redshift (z) of the galaxy to the halo's values
        self.lookbacktime = self.halo.lookbacktime
        self.z = self.halo.z
        self.age += timestep

        # update the time-variable quantities involved, in this case
        # GAR (and through it MIR and through the latter sMIR)
        # and sSFR (and through it rsSFR)
        self.update_GAR()
        self.update_sSFR(mode='from rsSFR')

        # (compute and) update current fractions of how inflows are distributed
        self.update_fstar(mode="beforehand")
        self.update_fout(mode="beforehand")
        self.update_fgas(mode="beforehand")

        # updating the change rates for stars, outflows and gas reservoir
        self.update_SFR(mode='from fstar')
        self.update_MLR(mode='from fout')
        self.update_GCR(mode='from fgas')

        # update the total gross mass accreted onto the galaxy
        self.update_macc(timestep=timestep)
        # update stellar mass, gas mass and ejected mass
        self.update_mstar(timestep=timestep, mode="from reduced SFR")
        self.update_mout(timestep=timestep)
        self.update_mgas(timestep=timestep)
        # also update total halo mass
        self.update_mhalo(timestep=timestep)

        return self.make_snapshot()


    # ------------------------------
    # time integration of the system - intuitive method without z-dependent sSFR as input

    def intuitive_evolve(self,
                         timestep: float = 1.e-3
                         ) -> 'Snapshot':
        """Evolves the galaxy by one timestep intuitively, *without* using
        sSFR(mstar, z) as time-dependent input (the latter is what Lilly+13 do)"""

        # # THIS IS ONLY NEEDED FOR THE AFTERWARD CALCULATION OF FRACTIONS (similar to Lilly) – commented out
        # # store snapshot of previous state before anything gets changed
        # self.previous = None
        # self.previous = self.make_snapshot()

        # update the lookbacktime and redshift (z) of the galaxy to the halo's values
        self.lookbacktime = self.halo.lookbacktime
        self.z = self.halo.z
        self.age += timestep

        # update the time-variable quantities involved, in this case
        # GAR (and through it MIR and through the latter sMIR)
        self.update_GAR()

        # updating the change rates for stars, outflows and gas reservoir
        self.update_SFR(mode='from SFE')
        self.update_MLR(mode='from MLF')
        self.update_GCR(mode='from GAR')

        # update the total gross mass accreted onto the galaxy
        self.update_macc(timestep=timestep)
        # update stellar mass, gas mass and ejected mass
        self.update_mstar(timestep=timestep, mode="from unreduced SFR")
        self.update_mout(timestep=timestep)
        self.update_mgas(timestep=timestep)
        # # also update total halo mass
        # self.update_mhalo(timestep=timestep)  # gets done now at Halo object level

        # update some derived quantities, here sSFR (and through it rsSFR)
        self.update_sSFR(mode='from SFR')

        # # THIS IS THE ONLY PART THAT NEEDS ALL THE ADDITIONAL SNAPSHOTS – commented out
        # # (compute and) update increases/decreases as fractions of gross accreted mass onto galaxy
        # self.update_fstar(mode="afterwards")
        # self.update_fout(mode="afterwards")
        # self.update_fgas(mode="afterwards")

        return self.make_snapshot()


    def evolve(self,
               mode: str = "intuitive",
               timestep: float = 1.e-3
               ) -> 'Snapshot':
        """
        Evolves the galaxy by one 'timestep' according to different prescriptions specified in 'mode'

        :param mode: either 'intuitive' or 'reference', calls respective [...]_evolve method
        :param timestep: length of timestep
        :return: snapshot of the galaxy
        """
        if mode == "reference":
            snp = self.reference_evolve(timestep=timestep)
        elif mode == "intuitive":
            snp = self.intuitive_evolve(timestep=timestep)
        else:
            raise ValueError("Unsupported keyword for 'mode' in Galaxy.evolve(). \n"
                             f"Galaxy '{self.name}' was not evolved")
        return snp


    # ----------------------------------------------------------------------------------
    # General functions usable for physical parameters that also might not change at all
    # C U R R E N T L Y   U N U S E D

    # Use with different values e.g. for SFE(mstar,t), mass loading fact(mgas,t)
    def Parameter_function(self, quant, t, quant0, t0, a1, a2, a3):
        """Some arbitrary function e.g. for star formation efficiency of form:
        offset + amplitude * (quant/quant_norm)**(factor * t/t_norm).
        Units: [1/time] + [1] * ([quant] / [quant])**([1] * [time] / [time])"""
        return a3 + a2 * ((quant/quant0)**(a1 * t/t0))

    # ∂ Parameter(Quantity,t) / ∂ t
    def pdv_Parameter_t_function(self, quant, t, quant0, t0, a1, a2, a3):
        """Partial derivative w.r.t. time t of the arbitrary function above"""
        return ((a1/t0) * np.log(quant/quant0) *
                (self.Parameter_function(quant, t, quant0, t0, a1, a2, a3) - a3))

    # ∂ Parameter(Quantity,t) / ∂ Quantity
    def pdv_Parameter_Quantity_function(self, quant, t, quant0, t0, a1, a2, a3):
        """Partial derivative w.r.t. the quantity of the arbitrary function above"""
        return ((a1*t)/(t0*quant) *
                (self.Parameter_function(quant, t, quant0, t0, a1, a2, a3) - a3))


    # ------------------------------------------
    # computing and updating physical quantities


    # fgas
    def update_fgas(self,
                    mode: str = "beforehand",
                    *args,
                    **kwargs
                    ) -> float:
        """Update the fraction of gas accreted onto the galaxy that goes into the reservoir"""
        if mode == "beforehand":
            self.fgas = self.compute_fgas_beforehand(*args, **kwargs)
        elif mode == "afterwards":
            self.fgas = self.compute_fgas_afterwards(*args, **kwargs)
        else:
            print("Unsupported keyword for 'mode' in Galaxy.update_fgas()."
                  "fgas not updated")
        return self.fgas

    def compute_fgas_beforehand(self,
                                IRF: float = None,
                                MLF: float = None,
                                SFE: float = None,
                                sSFR: float = None
                                ) -> float:
        """Fraction of incoming gas added to the galaxy gas reservoir,
        based on Eq. 14a from Lilly+13, for ideal regulator (fres)"""
        IRF = self.IRF if IRF is None else IRF
        MLF = self.MLF if MLF is None else MLF
        SFE = self.SFE if SFE is None else SFE
        sSFR = self.sSFR if sSFR is None else sSFR

        _tmp = (1.
                + MLF / (1. - IRF)
                + sSFR / SFE)
        return (sSFR / SFE) / _tmp

    def compute_fgas_afterwards(self,
                                delta_macc: float = None,
                                delta_mgas: float = None
                                ) -> float:
        """Fraction of incoming gas added to the galaxy gas reservoir,
        based on the mgas and macc increase"""
        delta_macc = (self.macc - self.previous.data["macc"]) if delta_macc is None else delta_macc
        delta_mgas = (self.mgas - self.previous.data["mgas"]) if delta_mgas is None else delta_mgas

        # return either the fraction, or - if denominator is zero, appropriate substitute to avoid div. by 0
        if delta_macc != 0:
            return delta_mgas / delta_macc
        elif delta_mgas == 0:  # assumes 0./0. to be zero (should be okay with regard to these fractions)
            return 0.
        else:
            return np.inf


    # fstar
    def update_fstar(self,
                     mode: str = "beforehand",
                     *args,
                     **kwargs
                     ) -> float:
        """Update the fraction of gas accreted onto the galaxy turned into long-lived stars"""
        if mode == "beforehand":
            self.fstar = self.compute_fstar_beforehand(*args, **kwargs)
        elif mode == "afterwards":
            self.fstar = self.compute_fstar_afterwards(*args, **kwargs)
        else:
            print("Unsupported keyword for 'mode' in Galaxy.update_fstar()."
                  "fstar not updated")
        return self.fstar

    def compute_fstar_beforehand(self,
                                 IRF: float = None,
                                 MLF: float = None,
                                 SFE: float = None,
                                 sSFR: float = None
                                 ) -> float:
        """Fraction of incoming gas converted to stars,
        based on Eq. 12a from Lilly+13, for ideal regulator"""
        IRF = self.IRF if IRF is None else IRF
        MLF = self.MLF if MLF is None else MLF
        SFE = self.SFE if SFE is None else SFE
        sSFR = self.sSFR if sSFR is None else sSFR

        _tmp = (1.
                + MLF / (1. - IRF)
                + sSFR / SFE)
        return 1./_tmp

    def compute_fstar_afterwards(self,
                                 delta_macc: float = None,
                                 delta_mstar: float = None
                                 ) -> float:
        """Fraction of incoming gas converted to (long lived) stars,
        based on the mstar and macc increase"""
        delta_macc = (self.macc - self.previous.data["macc"]) if delta_macc is None else delta_macc
        delta_mstar = (self.mstar - self.previous.data["mstar"]) if delta_mstar is None else delta_mstar

        # return either the fraction, or - if denominator is zero, appropriate substitute to avoid div. by 0
        if delta_macc != 0:
            return delta_mstar / delta_macc
        elif delta_mstar == 0:  # assumes 0./0. to be zero (should be okay with regard to these fractions)
            return 0.
        else:
            return np.inf


    # fout
    def update_fout(self,
                    mode: str = "beforehand",
                    *args,
                    **kwargs
                    ) -> float:
        """Update the fraction of gas accreted onto the galaxy expelled out of the galaxy again"""
        if mode == "beforehand":
            self.fout = self.compute_fout_beforehand(*args, **kwargs)
        elif mode == "afterwards":
            self.fout = self.compute_fout_afterwards(*args, **kwargs)
        else:
            print("Unsupported keyword for 'mode' in Galaxy.update_fout()."
                  "fout not updated")
        return self.fout

    def compute_fout_beforehand(self,
                                IRF: float = None,
                                MLF: float = None,
                                SFE: float = None,
                                sSFR: float = None
                                ) -> float:
        """Fraction of incoming gas expelled again from the galaxy,
        based on Eq. 13a from Lilly+13, for ideal regulator"""
        IRF = self.IRF if IRF is None else IRF
        MLF = self.MLF if MLF is None else MLF
        SFE = self.SFE if SFE is None else SFE
        sSFR = self.sSFR if sSFR is None else sSFR

        _tmp = (1.
                + MLF / (1. - IRF)
                + sSFR / SFE)
        return (MLF / (1. - IRF)) / _tmp

    def compute_fout_afterwards(self,
                                delta_macc: float = None,
                                delta_mout: float = None
                                ) -> float:
        """Fraction of incoming gas expelled again from the galaxy,
        based on the mout and macc increase"""
        delta_macc = (self.macc - self.previous.data["macc"]) if delta_macc is None else delta_macc
        delta_mout = (self.mout - self.previous.data["mout"]) if delta_mout is None else delta_mout

        # return either the fraction, or - if denominator is zero, appropriate substitute to avoid div. by 0
        if delta_macc != 0:
            return delta_mout / delta_macc
        elif delta_mout == 0:  # assumes 0./0. to be zero (should be okay with regard to these fractions)
            return 0.
        else:
            return np.inf


    # # gasmassfraction
    # def update_gasmassfraction(self, *args, **kwargs) -> float:
    #     self.gasmassfraction = self.compute_gasmassfraction(*args, **kwargs)
    #     return self.gasmassfraction
    #
    # def compute_gasmassfraction(self,
    #                             SFefficiency: float,
    #                             sSFR: float,
    #                             kappa: float = None,
    #                             mstar: float = None
    #                             ) -> float:
    #     """Compute the gas mass fraction according to Lilly et al. 2013,
    #     Eqns. (7) and (7a)"""
    #     gmf = sSFR / SFefficiency
    #     if (kappa is not None) and (mstar is not None):
    #         # this is Eq. (7a) ; with more information and dep. on mstar
    #         exp1 = 1. / kappa
    #         exp2 = - (kappa - 1.) / kappa
    #         return gmf**exp1 * mstar**exp2
    #     else:
    #         # this is Eq. (7) ; the simplified case
    #         return gmf


    # GAR
    def update_GAR(self, MIR: float = None, *args, **kwargs) -> float:
        """Updates the (absolute) gas accretion rate of the galaxy
        and calls compute_GAR() to get the new value. If the overall
        halo mass increase rate (MIR) is not provided, it also calls
        update_MIR() to compute the current value of that."""
        MIR = self.halo.MIR if MIR is None else MIR

        self.GAR = self.compute_GAR(MIR=MIR)
        return self.GAR

    def compute_GAR(self,
                    MIR: float = None,
                    BDR: float = None,
                    fgal: float = None
                    ) -> float:
        """Compute the absolute gas accretion rate of the galaxy
        from the overall halo mass increase rate (MIR) as well as
        the baryon-dark matter ratio (BDR) and the fraction of
        infalling gas that makes it deep enough into the galaxy
        to participate in star formation etc."""
        MIR = self.halo.MIR if MIR is None else MIR
        BDR = self.halo.BDR if BDR is None else BDR
        fgal = self.fgal if fgal is None else fgal

        return MIR * BDR * fgal


    # GCR
    def update_GCR(self,
                   mode:str = "from fgas",
                   *args,
                   **kwargs
                   ) -> float:
        """Update the gas change rate of the reservoir either intuitively, or based on fractions like Lilly+13"""
        if mode == "from fgas":
            self.GCR = self.compute_GCR_from_fgas(*args, **kwargs)
        elif mode == "from GAR":
            self.GCR = self.compute_GCR_from_GAR(*args, **kwargs)
        else:
            print(f"Unsupported keyword for 'mode' in Galaxy.update_GCR()."
                  f"GCR not updated")
        return self.GCR

    def compute_GCR_from_fgas(self,
                              GAR: float = None,
                              fgas: float = None
                              ) -> float:
        """Compute (reservoir) gas change rate from fractions in Lilly+13 ideal regulator"""
        GAR = self.GAR if GAR is None else GAR
        fgas = self.fgas if fgas is None else fgas

        return fgas * GAR

    def compute_GCR_from_GAR(self,
                             GAR: float = None,
                             IFR: float = None,
                             MLR: float = None,
                             SFR: float = None
                             ) -> float:
        """Compute (reservoir) gas change rate more 'intuitively' straight
        from overall accretion of gas into the galaxy, minus the gas that is
        expelled or turned into (long or short lived aka any kind of) stars.
        The instant return of gas from short lived stars is handled at the
        level of the integration of mstar, mgas (THE LAST BIT IS PROBABLY NOT TRUE ANYMORE!!!)"""
        GAR = self.GAR if GAR is None else GAR
        IRF = self.IRF if IFR is None else IFR
        MLR = self.MLR if MLR is None else MLR
        SFR = self.SFR if SFR is None else SFR

        return GAR - (MLR + (1 - IRF) * SFR)


    # macc
    def update_macc(self,
                    timestep: float,
                    *args,
                    **kwargs
                    ) -> float:
        """Update the total mass accreted onto the galaxy"""
        self.macc = self.compute_macc(timestep=timestep, *args, **kwargs)

        return self.macc

    def compute_macc(self,
                     timestep: float,
                     GAR: float = None,
                     macc: float = None
                     ) -> float:
        """Compute the new total mass accreted by the galaxy"""
        GAR = self.GAR if GAR is None else GAR
        macc = self.macc if macc is None else macc

        macc += (GAR * timestep)
        return macc if macc > 0. else 0.


    # mgas
    def update_mgas(self,
                    timestep: float,
                    *args,
                    **kwargs
                    ) -> float:
        """Update the gas mass (in reservoir)"""
        self.mgas = self.compute_mgas(timestep=timestep, *args, **kwargs)

        return self.mgas

    def compute_mgas(self,
                     timestep: float,
                     GCR: float = None,
                     mgas: float = None
                     ) -> float:
        """Compute the new gas mass (in reservoir) from GCR that already factors in a SFR reduced by (1 - IRF)"""
        GCR = self.GCR if GCR is None else GCR
        mgas = self.mgas if mgas is None else mgas

        mgas += (GCR * timestep)
        return mgas if mgas > 0. else 0.


    # # mhalo
    # def update_mhalo(self, timestep: float, *args, **kwargs) -> float:
    #     """Update the halo mass using the compute_mhalo() method"""
    #     self.mhalo = self.compute_mhalo(timestep, *args, **kwargs)
    #     return self.mhalo
    #
    # def compute_mhalo(self,
    #                   timestep: float,
    #                   HLF: float = None,
    #                   mhalo: float = None,
    #                   mout: float = None,
    #                   MIR: float = None,
    #                   ) -> float:
    #     """Compute new halo mass based on previous mhalo, mass increase rate
    #     and time step (integration), as well as precomputed (already integrated)
    #     baryonic mass loss from the galaxy into the halo and the fraction of
    #     which is also fully lost from the halo"""
    #     HLF = self.HLF if HLF is None else HLF
    #     mhalo = self.mhalo if mhalo is None else mhalo
    #     mout = self.mout if mout is None else mout
    #     MIR = self.MIR if MIR is None else MIR
    #
    #     return mhalo + (MIR * timestep) - (HLF * mout)


    # # MIR
    # def update_MIR(self, sMIR: float = None, *args, **kwargs) -> float:
    #     if sMIR is None:
    #         self.update_sMIR()
    #
    #     self.MIR = self.compute_MIR(*args, **kwargs)
    #     return self.MIR
    #
    # def compute_MIR(self,
    #                 mhalo: float = None,
    #                 sMIR: float = None
    #                 ) -> float:
    #     mhalo = self.mhalo if mhalo is None else mhalo
    #     sMIR = self.sMIR if sMIR is None else sMIR
    #
    #     return sMIR * mhalo


    # MLR
    def update_MLR(self,
                   mode:str = "from fout",
                   *args,
                   **kwargs
                   ) -> float:
        """Update the mass loss rate, which is the rate of mass that gets ejected from the galaxy"""
        if mode == "from fout":
            self.MLR = self.compute_MLR_from_fout(*args, **kwargs)
        elif mode == "from MLF":
            self.MLR = self.compute_MLR_from_MLF(*args, **kwargs)
        else:
            print(f"Unsupported keyword for 'mode' in Galaxy.update_MLR()."
                  f"MLR not updated")
        return self.MLR

    def compute_MLR_from_fout(self,
                              GAR: float = None,
                              fout: float = None
                              ) -> float:
        """Compute mass loss rate from fractions in Lilly+13 ideal regulator"""
        GAR = self.GAR if GAR is None else GAR
        fout = self.fout if fout is None else fout

        return fout * GAR

    def compute_MLR_from_MLF(self,
                             MLF: float = None,
                             SFR: float = None
                             ) -> float:
        """Compute mass loss rate 'intuitively' from mass-loading factor
        and star formation rate"""
        MLF = self.MLF if MLF is None else MLF
        SFR = self.SFR if SFR is None else SFR

        return MLF * SFR


    # mout
    def update_mout(self, timestep: float, *args, **kwargs) -> float:
        """Update the total mass gas lost from the system up to this point"""
        self.mout = self.compute_mout(timestep=timestep, *args, **kwargs)
        return self.mout

    def compute_mout(self,
                     timestep: float,
                     MLR: float = None,
                     mout: float = None
                     ) -> float:
        """Compute the total gas mass lost from the system until now"""
        MLR = self.MLR if MLR is None else MLR
        mout = self.mout if mout is None else mout

        mout += (MLR * timestep)
        return mout if mout > 0. else 0.


    # mstar
    def update_mstar(self,
                     timestep: float,
                     mode: str = "from reduced SFR",
                     *args,
                     **kwargs
                     ) -> float:
        """Update current stellar mass according to the 'mode' keyword provided"""
        if mode == "from reduced SFR":
            self.mstar = self.compute_mstar_from_reduced_SFR(timestep=timestep, *args, **kwargs)
        elif mode == "from unreduced SFR":
            self.mstar = self.compute_mstar_from_unreduced_SFR(timestep=timestep, *args, **kwargs)
        else:
            print("Unsupported keyword for 'mode' in Galaxy.update_mstar()."
                  "mstar not updated")
        return self.mstar

    def compute_mstar_from_reduced_SFR(self,
                                       timestep: float,
                                       mstar: float = None,
                                       SFR: float = None
                                       ) -> float:
        """Compute current stellar mass from SFR already reduced by (1 - IRF)"""
        mstar = self.mstar if mstar is None else mstar
        SFR = self.SFR if SFR is None else SFR

        mstar += (SFR * timestep)
        return mstar if mstar > 0. else 0.

    def compute_mstar_from_unreduced_SFR(self,
                                         timestep: float,
                                         IRF: float = None,
                                         mstar: float = None,
                                         SFR: float = None
                                         ) -> float:
        """Compute current stellar mass from SFR unreduced by (1 - IRF)"""
        IRF = self.IRF if IRF is None else IRF
        mstar = self.mstar if mstar is None else mstar
        SFR = self.SFR if SFR is None else SFR

        mstar += ((1 - IRF) * SFR * timestep)
        return mstar if mstar > 0. else 0.

    # rsSFR
    def update_rsSFR(self,
                     mode: str = 'empirical',
                     *args,
                     **kwargs
                     ) -> float:
        """Update the reduced specific star-formation rate"""
        if mode == 'empirical':
            self.rsSFR = self.compute_rsSFR_empirical(*args, **kwargs)
        elif mode == 'from sSFR':
            self.rsSFR = self.compute_rsSFR_from_sSFR(*args, **kwargs)
        else:
            print("Unsupported keyword for 'mode' in Galaxy.update_rsSFR()."
                  "rsSFR not updated")
        return self.rsSFR

    def compute_rsSFR_empirical(self,
                                mstar: float = None,
                                z: float = None
                                ) -> float:
        """Computes the reduced specific star formation rate according to
        Lilly et al. 2013, Eq. (2), for z < 2. and z > 2., respectively"""
        mstar = self.mstar if mstar is None else mstar
        z = self.z if z is None else z

        if z < 2.:
            return 0.07 * (10**10.5 / mstar)**(0.1) * (1 + z)**3 #* 10**(-9)
        else:
            return 0.3 * (10**10.5 / mstar)**(0.1) * (1 + z)**(5/3) #* 10**(-9)

    def compute_rsSFR_from_sSFR(self,
                                IRF: float = None,
                                sSFR:float = None) -> float:
        """Computes the reduced specific star formation rate from
        the non-reduced, regular specific star formation rate"""
        IRF = self.IRF if IRF is None else IRF
        sSFR = self.sSFR if sSFR is None else sSFR

        return (1. - IRF) * sSFR


    # SFR
    def update_SFR(self,
                   mode: str = 'from fstar',
                   *args,
                   **kwargs
                   ) -> float:
        """Compute and update SFR according to 'mode' keyword"""
        if mode == 'from fstar':
            self.SFR = self.compute_SFR_from_fstar(*args, **kwargs)
        elif mode == 'from SFE':
            self.SFR = self.compute_SFR_from_SFE(*args, **kwargs)
        else:
            print("Unsupported keyword for 'mode' in Galaxy.update_SFR()."
                  "SFR not updated")
        return self.SFR

    def compute_SFR_from_fstar(self,
                               GAR: float = None,
                               fstar: float = None
                               ) -> float:
        """Compute star formation rate from fractions in Lilly+13 ideal regulator"""
        GAR = self.GAR if GAR is None else GAR
        fstar = self.fstar if fstar is None else fstar

        return fstar * GAR

    def compute_SFR_from_SFE(self,
                             mgas: float = None,
                             SFE: float = None
                             ) -> float:
        """Compute star formation rate intuitively from star formation efficiency
        and available gas"""
        mgas = self.mgas if mgas is None else mgas
        SFE = self.SFE if SFE is None else SFE

        return SFE * mgas


    # # sMIR
    # def update_sMIR(self, *args, **kwargs) -> float:
    #     self.sMIR = self.compute_sMIR(*args, **kwargs)
    #     return self.sMIR
    #
    # def compute_sMIR(self,
    #                  mhalo: float = None,
    #                  z: float = None
    #                  ) -> float:
    #     """Computes the specific Mass Increase Rate of the DM halo
    #     accoding to Lilly et al. 2013, Eq. (3), more precise version"""
    #     mhalo = self.mhalo if mhalo is None else mhalo
    #     z = self.z if z is None else z
    #
    #     return 0.027 * (mhalo / 10**12)**(0.15) * (1 + z + 0.1*((1 + z)**(-1.25)))**2.5


    # sSFR
    def update_sSFR(self,
                    mode: str = 'from rsSFR',
                    rsSFR: float = None,
                    *args,
                    **kwargs
                    ) -> float:
        """Updates the actual, unreduced sSFR and, if not supplied, the (reduced) rsSFR too"""

        if mode == 'from rsSFR':
            # following Lilly+13 ideal regulator
            # update rsSFR first if not supplied, since it's based off it
            if rsSFR is None:
                self.update_rsSFR(mode='empirical')
            # now compute and update sSFR
            self.sSFR = self.compute_sSFR_from_rsSFR(*args, **kwargs)
        elif mode == 'from SFR':
            # following the 'intuitive' approach and derive it from SFR and mstar
            self.sSFR = self.compute_sSFR_from_SFR(*args, **kwargs)
            # also update dependent quantity, rsSFR, after the fact
            if rsSFR is None:
                self.update_rsSFR(mode='from sSFR')
            else:
                print("Unsupported keyword for 'mode' in Galaxy.update_sSFR()."
                      "sSFR not updated")

        return self.sSFR

    def compute_sSFR_from_rsSFR(self,
                                rsSFR: float = None,
                                IRF: float = None
                                ) -> float:
        """Computes the actual, unreduced sSFR from the rsSFR"""
        rsSFR = self.rsSFR if rsSFR is None else rsSFR
        IRF = self.IRF if IRF is None else IRF

        return rsSFR / (1. - IRF)

    def compute_sSFR_from_SFR(self,
                              mstar: float = None,
                              SFR: float = None
                              ) -> float:
        """Computes the intuitive sSFR from SFR and mstar"""
        mstar = self.mstar if mstar is None else mstar
        SFR = self.SFR if SFR is None else SFR

        return SFR / mstar
