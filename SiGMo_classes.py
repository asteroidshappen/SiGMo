# =======
# Imports

# basic
import numpy as np

# astronomy
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value


class Environment:
    """Class for simple environment of one or more galaxies of the Galaxies
    class.

    Attributes
    ----------
    age : float, optional
        The current age of the system in Gyrs (default 0.)
    galaxies : list, optional
        List of all Galaxy objects in this environment (default None)
    lookbacktime : float, optional
        The current cosmic lookback time (default None)
    mgas : float, optional
        The amount of gas available in the environment (default np.inf)
    name : str, optional
        The name of the galaxy (default 'Test_Env')
    z : float, optional
        The current redshift of the system (default None)
    zstart : float, optional
        The initial redshift of the system (default 6.)

    Methods
    -------
    create_Galaxy(galaxy_kwargs: dict = None)
        Creates Galaxy object and adds it to galaxies list
    reference_evolve(timestep: float = 1.e-3)
        Evolves the environment and all galaxies acc. to Lilly+13, Eq.12a,13a,14a
    """

    def __init__(self,
                 age: float = 0.,
                 galaxies: list = None,
                 lookbacktime: float = None,
                 mgas: float = np.inf,
                 name: str = 'Test_Env',
                 zstart: float = 6.
                 ):
        self.age = age
        self.galaxies = galaxies if galaxies else []
        self.lookbacktime = lookbacktime
        self.mgas = mgas
        self.name = name
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
                      f"Will use the supllied value of z={zstart:9.2f} to calculate\n" +
                      f"the corresponding lookback time from it.")
            self.lookbacktime = cosmo.lookback_time(self.zstart).value  # IMPLICIT HARDCODING OF Gyr from astropy routine!

        # compute zcurrent from self.zstart
        self.z = self.zstart
        return

    def __repr__(self):
        r_string = ", ".join("=".join((str(k),(f"List of {len(v)} Galaxy() objects{'' if len(v) == 0 else ': '}" + ', '.join(f"'{g.name}'" for g in v) if k == 'galaxies' else str(v)))) for k,v in vars(self).items())
        return f'Environment({r_string})'

    def __str__(self):
        s_string = "\n".join(" = ".join(("  " + str(k),(f"List of {len(v)} Galaxy() objects{'' if len(v) == 0 else ': '}" + ', '.join(f"'{g.name}'" for g in v) if k == 'galaxies' else str(v)))) for k,v in vars(self).items())
        return f'Instance of Environment() with the following attributes:\n' + s_string


    def create_Galaxy(self,
                      **galaxy_kwargs: dict
                      ):
        gal = Galaxy(env=self, **galaxy_kwargs)
        self.galaxies.append(gal)
        return gal


    def reference_evolve(self,
                         timestep: float = 1.e-3
                         ):
        # make the time step in lookbacktime, then convert to z as well
        self.lookbacktime -= timestep
        self.z = z_at_value(cosmo.age, cosmo.age(0) - self.lookbacktime * u.Gyr) #, zmin=self.z-.5, zmax=self.z+.5)  # Gyr HARDCODED AGAIN!

        # go through all the galaxies and evolve/update them based on new time
        for gal in self.galaxies:
            gal.reference_evolve(timestep=timestep)
        return



# ========================================================



class Galaxy:
    """Class for simple model of a galaxy. Every galaxy should be associated
    with exactly one Environment object, which provides a method to create
    a galaxy and add it to its galaxy list.

    Attributes
    ----------
    env : Environment
        The environment object the galaxy is associated with/located in
    accretionrate : float, optional
        The accretion rate of galaxy (default 1.)
    BDR : float, optional
        The ratio of (gaseous) baryonic to dark matter entering the halo (default 0.2)
    fgal : float, optional
        The fraction of baryons that enter the halo and make it all the way down
        into the "regulator system" to participate in star formation etc
    GCR : float, optional
        The change rate ∂/∂t in gas mass content of the galaxy (default 0.)
    IRF: float, optional
        The fraction of gas being converted to stars that is promptly,
        here instantly, returned to the gas reservoir (default 0.4)
    MLF : float, optional
        The mass-loading factor coupling SFR and mass loss (default 0.1)
    MLR : float, optional
        The mass loss rate from the galaxy (default 0.)
    mgas : float, optional
        The gas mass content of the galaxy (default 1.e10)
    mhalo : float, optional
        The total mass of the halo that the galaxy resides in (default 1.e12)
    mstar : float, optional
        The stellar mass content of the galaxy (default 1.e9)
    mout: float, optional
        The mass lost from the system due to massloss (default 0.)
    name : str, optional
        The name of the galaxy (default 'Test_Gal')
    rsSFR : float, optional
        The reduced specific SFR; excludes instant.returned gas (default 0.)
    SFE : float, optional
        The star formation efficiency (default 0.3)
    SFR : float, optional
        The star formation rate in the galaxy (default 0.)
    sMIR : float, optional
        The specific Mass Increase Rate (accretion) of the DM halo (default 0.)
    sSFR: float, optional
        The actual specific star formation rate; this sSFR does not account
        for reduction by the inst. return to the gas reservoir (default 0.)
    z: float, optional
        The current redshift of the galaxy, gets it from Environment (default None)

    Methods
    -------
    compute_fstar(self) -> float:
        Fraction of incoming gas converted to stars,
        based on Eq. 12a from Lilly+13, for ideal regulator
    compute_fout(self) -> float:
        Fraction of incoming gas expelled again from the galaxy,
        based on Eq. 13a from Lilly+13, for ideal regulator
    compute_fgas(self) -> float:
        Fraction of incoming gas added to the galaxy gas reservoir,
        based on Eq. 14a from Lilly+13, for ideal regulator
    reference_evolve(timestep: float = 1.e-3)
        Evolves the galaxy by one timestep according to Lilly+13 Eq.12a,13a,14a,
        ideal regulator
    """

    def __init__(self,
                 env: Environment,
                 accretionrate: float = 1.,
                 BDR: float = 0.2,
                 fgal: float = 1.,
                 fgas: float = None,
                 fout: float = None,
                 fstar: float = None,
                 gasmassfraction: float = 1.,
                 GCR: float = 0.,
                 IRF: float = 0.4,
                 MLF: float = 0.1,
                 MLR: float = 0.,
                 mgas: float = 1.e10,
                 mhalo: float = 1.e12,
                 mstar: float = 1.e9,
                 mout: float = 0.,
                 name: str = 'Test_Gal',
                 rsSFR: float = 0.,
                 SFE: float = 0.3,
                 SFR: float = 0.,
                 sMIR: float = 0.,
                 sSFR: float = 0.,
                 z: float = None
                 ):
        self.env = env
        self.accretionrate = accretionrate
        self.BDR = BDR
        self.fgal = fgal
        self.fgas = fgas
        self.fout = fout
        self.fstar =fstar
        self.GCR = GCR
        self.IRF = IRF
        self.MLF = MLF
        self.MLR = MLR
        self.mgas = mgas
        self.mhalo = mhalo
        self.mstar = mstar
        self.mout = mout
        self.name = name
        self.rsSFR = rsSFR
        self.SFE = SFE
        self.SFR = SFR
        self.sMIR = sMIR
        self.sSFR = sSFR
        self.z = env.z if z is None else z

    def __repr__(self):
        r_string = ", ".join("=".join((str(k),(f"Environment() object '{v.name}'" if k == 'env' else str(v)))) for k,v in vars(self).items())
        return f'Galaxy({r_string})'

    def __str__(self):
        s_string = "\n".join(" = ".join(("  " + str(k),(f"Environment() object '{v.name}'" if k == 'env' else str(v)))) for k,v in vars(self).items())
        return f'Instance of Galaxy() with the following attributes:\n' + s_string


    # ------------------------------
    # time integration of the system

    # This is the reference model from Lilly et al. 2013, ideal case
    # based on Eqns. (12a), (13a), (14a)
    # fractional splitting of incoming baryons, cf.Lilly+13,Eqns. 12-14a

    # fstar
    def update_fstar(self, *args, **kwargs) -> float:
        self.fstar = self.compute_fstar(*args, **kwargs)
        return self.fstar

    def compute_fstar(self,
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

    # fout
    def update_fout(self, *args, **kwargs) -> float:
        self.fout = self.compute_fout(*args, **kwargs)
        return self.fout

    def compute_fout(self,
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

    # fgas
    def update_fgas(self, *args, **kwargs) -> float:
        self.fgas = self.compute_fgas(*args, **kwargs)
        return self.fgas

    def compute_fgas(self,
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


    def reference_evolve(self,
                         timestep: float = 1.e-3
                         ):
        """Evolves the galaxy by one timestep according to Lilly+13 Eq.12a,13a,14a,
        ideal regulator"""
        # update the redshift of the galaxy to the environment's z
        self.z = self.env.z

        # update the time-variable quantities involved, in this case
        # accretionrate (and through is sMIR) and sSFR (and through it rsSFR)
        self.update_accretionrate()
        self.update_sSFR()

        # (compute and) update current fractions of how inflows are distributed
        self.update_fstar()
        self.update_fout()
        self.update_fgas()

        # updating the change rates for stars, outflows and gas reservoir
        self.update_SFR()
        self.update_MLR()
        self.update_GCR()

        # update stellar mass, gas mass and ejected mass
        self.update_mstar(timestep=timestep)
        self.update_mout(timestep=timestep)
        self.update_mgas(timestep=timestep)
        # ===================
        # UPDATING mhalo GOES HERE !!!
        # ===================

        return


    def make_snapshot(self):
        """Returns the current values of all major attributes as dict"""
        _tmp_out = dict(vars(self))
        _tmp_out['env'] = _tmp_out['env'].name if _tmp_out['env'] is not None else None
        return _tmp_out


    # -------------------------------------------------------------------
    # Functions for physical parameters that might also not change at all

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


    # accretionrate
    def update_accretionrate(self, sMIR: float = None, *args, **kwargs) -> float:
        if sMIR is None:
            self.update_sMIR()

        self.accretionrate = self.compute_accretionrate()
        return self.accretionrate

    def compute_accretionrate(self,
                              sMIR: float = None,
                              BDR: float = None,
                              fgal: float = None
                              ) -> float:
        sMIR = self.sMIR if sMIR is None else sMIR
        BDR = self.BDR if BDR is None else BDR
        fgal = self.fgal if fgal is None else fgal

        return sMIR * BDR * fgal


    # GCR
    def update_GCR(self, *args, **kwargs) -> float:
        self.GCR = self.compute_GCR(*args, **kwargs)
        return self.GCR

    def compute_GCR(self,
                    accretionrate: float = None,
                    fgas: float = None
                    ) -> float:
        """Compute (reservoir) gas change rate from fractions in Lilly+13 ideal regulator"""
        accretionrate = self.accretionrate if accretionrate is None else accretionrate
        fgas = self.fgas if fgas is None else fgas

        return fgas * accretionrate


    # mgas
    def update_mgas(self, timestep: float, *args, **kwargs) -> float:
        self.mgas = self.compute_mgas(timestep=timestep, *args, **kwargs)
        return self.mgas

    def compute_mgas(self,
                      timestep: float,
                      GCR: float = None,) -> float:
        GCR = self.GCR if GCR is None else GCR

        return GCR * timestep


    # MLR
    def update_MLR(self, *args, **kwargs) -> float:
        self.MLR = self.compute_MLR(*args, **kwargs)
        return self.MLR

    def compute_MLR(self,
                    accretionrate: float = None,
                    fout: float = None
                    ) -> float:
        """Compute mass loss rate from fractions in Lilly+13 ideal regulator"""
        accretionrate = self.accretionrate if accretionrate is None else accretionrate
        fout = self.fout if fout is None else fout

        return fout * accretionrate


    # mout
    def update_mout(self, timestep: float, *args, **kwargs) -> float:
        self.mout = self.compute_mout(timestep=timestep, *args, **kwargs)
        return self.mout

    def compute_mout(self,
                      timestep: float,
                      MLR: float = None,) -> float:
        MLR = self.MLR if MLR is None else MLR

        return MLR * timestep


    # mstar
    def update_mstar(self, timestep: float, *args, **kwargs) -> float:
        self.mstar = self.compute_mstar(timestep=timestep, *args, **kwargs)
        return self.mstar

    def compute_mstar(self,
                      timestep: float,
                      SFR: float = None,) -> float:
        SFR = self.SFR if SFR is None else SFR

        return SFR * timestep

    # rsSFR
    def update_rsSFR(self, *args, **kwargs) -> float:
        self.rsSFR = self.compute_rsSFR(*args, **kwargs)
        return self.rsSFR

    def compute_rsSFR(self,
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


    # SFR
    def update_SFR(self, *args, **kwargs) -> float:
        self.SFR = self.compute_SFR(*args, **kwargs)
        return self.accretionrate

    def compute_SFR(self,
                    accretionrate: float = None,
                    fstar: float = None
                    ) -> float:
        """Compute star formation rate from fractions in Lilly+13 ideal regulator"""
        accretionrate = self.accretionrate if accretionrate is None else accretionrate
        fstar = self.fstar if fstar is None else fstar

        return fstar * accretionrate

    # sMIR
    def update_sMIR(self, *args, **kwargs) -> float:
        self.sMIR = self.compute_sMIR(*args, **kwargs)
        return self.sMIR

    def compute_sMIR(self,
                     mhalo: float = None,
                     z: float = None
                     ) -> float:
        """Computes the specific Mass Increase Rate of the DM halo
        accoding to Lilly et al. 2013, Eq. (3), more precise version"""
        mhalo = self.mhalo if mhalo is None else mhalo
        z = self.z if z is None else z

        return 0.027 * (mhalo / 10**12)**(0.15) * (1 + z + 0.1*((1 + z)**(-1.25)))**2.5


    # sSFR
    def update_sSFR(self, rsSFR: float = None, *args, **kwargs) -> float:
        """Updates the actual, unreduced SFR and, if not supplied, the (reduced) rsSFR too"""
        # update rsSFR first if not supplied, since it's based off it
        if rsSFR is None:
            self.update_rsSFR()

        # now compute and update sSFR
        self.sSFR = self.compute_sSFR(*args, **kwargs)
        return self.sSFR

    def compute_sSFR(self,
                     rsSFR: float = None,
                     IRF: float = None
                     ) -> float:
        """Computes the actual, unreduced SFR from the rsSFR"""
        rsSFR = self.rsSFR if rsSFR is None else rsSFR
        IRF = self.IRF if IRF is None else IRF

        return rsSFR / (1. - IRF)

