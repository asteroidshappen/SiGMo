# Contains abstract Integrator base class with subclasses for different use-cases

# ======
# Import

# basic
import numpy as np
import warnings

# I/O
from pathlib import Path

# OOP
from abc import ABC, abstractmethod

# misc
from tqdm import trange


# ==================================================
# Abstract Base Class for all subsequent integrators

class Integrator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def integrate(self):
        pass

    def determine_dt(self,
                     dt_current: float,
                     dt_min: float = None,
                     dt_max: float = None,
                     dx_current: float = None,
                     dxdt_target: float = None):
        """Determine the timesteps. Currently two modes: (a) constant; (b) adaptive"""

        # constant timesteps
        if dxdt_target is None:
            dt_next = dt_current

        # adaptive timesteps (based on last step: dx= abs(x_0 - x_-1) , dt= length of timestep from t_-1 to t_0)
        else:
            dt_next = dx_current / dxdt_target
            if (dt_min is not None) and (dt_next < dt_min):
                dt_next = dt_min
            elif (dt_max is not None) and (dt_next > dt_max):
                dt_next = dt_max
        return dt_next


# =========================
# Fixed Timestep Integrator

class FTI(Integrator):
    """Fixed timestep integrator

    t_start, t_end are in 'lookbacktime' in Gyr, so a positive t_start=0.8 coresponds to 800 Myr in the past
    """
    def __init__(self,
                 env,
                 evolve_method = 'evolve',
                 dt: float = None,
                 t_start = None,
                 t_end = None,
                 n_steps: int = None
                 ):
        """
        Initialises an instance of the FTI (Fixed Timestep Integrator) class.
        Not all sensible input combinations are supported yet, e.g. t_start + dt + n_steps should work, but doesn't!

        :param env: Environment object that will be integrated/iterated over in time. env contains a list of all its
            Halos, which in turn each contain a list of their respective galaxies
        :param evolve_method: method of env that will be used to integrate env (Default: 'evolve')
        :param dt: time step size for the integration in Gyrs
        :param t_start: start time for the integration in Gyrs, as look-back time from present day
        :param t_end: end time for the integration in Gyrs, as look-back time from present day
        :param n_steps: number of steps for the integration

        :raises TypeError: if not sufficient time arguments (valid combinations of dt, t_start, t_end, n_steps)
            are provided to determine the rest of the time arguments
        :raises Warning: if both dt and n_steps are provided in combination with t_end, the time step size dt
            will be preferred and the number of integration steps n_steps will be recalculated accordingly
        """
        self.dt, self.n_steps, self.t_start, self.t_end = self.determine_time_attributes(dt, n_steps, t_end, t_start)

        # set object attributes
        self.env = env
        self.evolve_method = evolve_method if callable(evolve_method) else getattr(self.env, evolve_method)

        return

    def determine_time_attributes(self, dt, n_steps, t_end, t_start):
        """Deals with different time and step input combinations. Returns a completed set of time parameters,
        if possible.
        """
        # Could be extended in the future to allow for automatic end time: t_end = 0 if (t_end is None) and t_start
        # but where to put that without breaking the other conditions?
        err_str = f"{self.__class__.__name__}() missing required arguments: "
        t_start_str = ", t_start"
        if not any([dt, t_end is not None, n_steps]):
            raise TypeError(err_str, f'dt OR n_steps, t_end{t_start_str if t_start is None else ""}')
        elif t_end is None:
            raise TypeError(err_str, f't_end{t_start_str if t_start is None else ""}')
        elif not any([dt is None, t_end is None]) and not n_steps:
            t_start = t_start if t_start else 0.
            n_steps = int(np.floor((t_start - t_end) / dt))
            # n_steps = int(np.floor((t_start - t_end) / dt))
        elif not any([n_steps is None, t_end is None]) and not dt:
            t_start = t_start if t_start else 0.
            dt = (t_start - t_end) / n_steps
        elif not any([dt is None, n_steps is None]):
            t_start = t_start if t_start else 0.
            n_steps_prev = n_steps
            n_steps = int(np.floor((t_start - t_end) / dt))
            warnings.warn("Both dt and n_steps have been specified. dt is preferred. Setting n_steps to value "
                          f"inferred from dt; new n_steps={n_steps} (previously: {n_steps_prev})")
        return dt, n_steps, t_start, t_end

    def integrate(self,
                  wtd=1,
                  outdir='outputs/_tmp/',
                  single_snapshots: bool = True,
                  event_list = None
                  ):
        """
        The core integrator/iterator loop that includes writing Snapshot files to disk

        :param wtd: 'write to disk' frequency, 1 being every timestep and n being every n-th timestep (Default: 1)
        :param outdir: file path to the output directory, either as a string or as a pathlib Path object
            (Default: 'outputs/_tmp/')
        :return: Environment object that has been iterated over, after iteration is complete
        """
        # avoid repeated lookups
        env = self.env
        halo = env.halos[0]   # only works for the 1st halo (HARDCODED!)
        gal = halo.galaxies[0]   # only works for the 1st galaxy (HARDCODED!)
        evolve_method = self.evolve_method
        dt = self.dt
        n_steps = self.n_steps

        # turn outdir into pathlib object, if necessary
        if not isinstance(outdir, Path):
            if isinstance(outdir, str):
                _tmppath = Path(outdir)
                if _tmppath.is_dir():
                    outdir = _tmppath
                else:
                    _tmppath = Path.cwd() / _tmppath
                    if _tmppath.is_dir():
                        outdir = _tmppath
                    else:
                        raise FileNotFoundError("'outdir' could not be evaluated to an existing directory to use "
                                                f"for output: {outdir!r}")
            else:
                raise TypeError(f"'outdir' does not meet type requirements: {outdir!r}\n"
                                "It needs to be either a pathlib Path or a str representing the output directory")

        # create last subdir in path in case it doesn't exist
        outdir.mkdir(parents=False, exist_ok=True)

        # initial snapshots before first evolve even (basically ICs converted to AstroObjects, then right into Snapshot
        # write the Environment and all its Halos and Galaxies to disk
        i_write = 0
        env.make_and_write_all_snapshots(i_write, n_steps, outdir, single_snapshots=single_snapshots)

        # loop over time
        for t in trange(1, n_steps):
            # evolve the Environment and everything in it (env->halos->galaxies)
            evolve_method(timestep=dt)

            # check for scripted event
            if event_list is not None:
                while t == event_list[0][0]:
                    # take and remove first element of the envent_list. The order of items in the 2-d nested list is:
                    # number of timestep, Environment/Halo/Galaxy, attribute name, method of modif., value of modif.
                    _i, _obj_type, _quantity_name, _method, _value = event_list.pop(0)

                    # grab the right object (which will in turn have the quantity
                    if _obj_type.casefold() == "env".casefold():
                        _obj = env
                    elif _obj_type.casefold() == "halo".casefold():
                        _obj = halo
                    elif _obj_type.casefold() == "gal".casefold():
                        _obj = gal
                    else:
                        raise ValueError('an object to be modified by event_list does not exist or is called wrong')

                    # get the property, it should be used in all but one scenario
                    _quantity = getattr(_obj, _quantity_name)

                    # now do the modifications - careful, there's different scenarios
                    if _method.casefold() == 'set'.casefold():
                        setattr(_obj, _quantity_name, _value)
                    elif _method.casefold() == 'multiply'.casefold():
                        setattr(_obj, _quantity_name, _quantity * _value)
                    elif _method.casefold() == 'divide'.casefold():
                        setattr(_obj, _quantity_name, _quantity / _value)
                    elif _method.casefold() == 'add'.casefold():
                        setattr(_obj, _quantity_name, _quantity + _value)
                    elif _method.casefold() == 'subtract'.casefold():
                        setattr(_obj, _quantity_name, _quantity - _value)
                    else:
                        raise ValueError('a method for modifying by event_list does not exist or was selected wrong')

            # only go through snapshot creation if they will be written to disk
            if (wtd > 0) and ((t % wtd == 0) or (t == n_steps)):
                # write the Environment and all its Halos and Galaxies to disk
                i_write += 1
                env.make_and_write_all_snapshots(i_write, n_steps, outdir, single_snapshots=single_snapshots)

        return env


# ============================
# Variable Timestep Integrator

class VTI(Integrator):
    """Variable timestep integrator"""
    pass