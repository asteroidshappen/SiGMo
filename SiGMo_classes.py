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
    BDR : float, optional
        The ratio of (gaseous) baryonic to dark matter entering the halo (default 0.2)
    fgal : float, optional
        The fraction of baryons that enter the halo and make it all the way down
        into the "regulator system" to participate in star formation etc
    GAR : float, optional
        The gas accretion rate of the galaxy (default 1.e12)
    GCR : float, optional
        The change rate ∂/∂t in gas mass content of the galaxy (default 0.)
    HLF : float, optional
        The halo loss fraction - fraction of baryons that get expelled by feedback
        not only from the galaxy but also from the galaxy together (default 0.1)
    IRF: float, optional
        The fraction of gas being converted to stars that is promptly,
        here instantly, returned to the gas reservoir (default 0.4)
    MIR : float, optional
        The mass increase rate (accretion) of the DM halo with mass mhalo (default 0.)
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
        The specific mass increase rate (accretion) of the DM halo (default 0.)
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
                 BDR: float = 0.2,
                 fgal: float = 1.,
                 fgas: float = None,
                 fout: float = None,
                 fstar: float = None,
                 GAR: float = 1.e12,
                 gasmassfraction: float = 1.,
                 GCR: float = 0.,
                 HLF: float = 0.1,
                 IRF: float = 0.4,
                 MIR: float = 0.,
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
        self.BDR = BDR
        self.fgal = fgal
        self.fgas = fgas
        self.fout = fout
        self.fstar =fstar
        self.GAR = GAR
        self.GCR = GCR
        self.HLF = HLF
        self.IRF = IRF
        self.MIR = MIR
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
    # time integration of the system (reference method by Lilly+13, "ideal regulator")

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
        # GAR (and through it MIR and through the latter sMIR)
        # and sSFR (and through it rsSFR)
        self.update_GAR()
        self.update_sSFR(mode='from rsSFR')

        # (compute and) update current fractions of how inflows are distributed
        self.update_fstar()
        self.update_fout()
        self.update_fgas()

        # updating the change rates for stars, outflows and gas reservoir
        self.update_SFR(mode='from fstar')
        self.update_MLR(mode='from fout')
        self.update_GCR(mode='from fgas')

        # update stellar mass, gas mass and ejected mass
        self.update_mstar(timestep=timestep)
        self.update_mout(timestep=timestep)
        self.update_mgas(timestep=timestep)
        # also update total halo mass
        self.update_mhalo(timestep=timestep)

        return


    # ------------------------------
    # time integration of the system - intuitive method without z-dependent sSFR as input

    def intuitive_evolve(self,
                         timestep: float = 1.e-3
                         ):
        """Evolves the galaxy by one timestep intuitively, *without* using
        sSFR(mstar, z) as time-dependent input (like Lilly+13 do)"""
        # update the redshift of the galaxy to the environment's z
        self.z = self.env.z

        # update the time-variable quantities involved, in this case
        # GAR (and through it MIR and through the latter sMIR)
        self.update_GAR()


        # FRACTIONS fstar, fout, fgas NOT NEEDED IN INTUITIVE MODEL ANYMORE -
        # maybe compute afterwards for comparison with 'reference_evolve()'?
        # # (compute and) update current fractions of how inflows are distributed
        # self.update_fstar()
        # self.update_fout()
        # self.update_fgas()

        # updating the change rates for stars, outflows and gas reservoir
        self.update_SFR(mode='from SFE')
        self.update_MLR(mode='from MLF')
        self.update_GCR(mode='from GAR')

        # update stellar mass, gas mass and ejected mass
        self.update_mstar(timestep=timestep)
        self.update_mout(timestep=timestep)
        self.update_mgas(timestep=timestep)
        # also update total halo mass
        self.update_mhalo(timestep=timestep)

        # update some derived quantities, here sSFR (and through it rsSFR)
        self.update_sSFR(mode='from SFR')

        return





    # make snapshot of the system

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


    # GAR
    def update_GAR(self, MIR: float = None, *args, **kwargs) -> float:
        """Updates the (absolute) gas accretion rate of the galaxy
        and calls compute_GAR() to get the new value. If the overall
        halo mass increase rate (MIR) is not provided, it also calls
        update_MIR() to compute the current value of that."""
        if MIR is None:
            self.update_MIR()

        self.GAR = self.compute_GAR()
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
        MIR = self.MIR if MIR is None else MIR
        BDR = self.BDR if BDR is None else BDR
        fgal = self.fgal if fgal is None else fgal

        return MIR * BDR * fgal


    # GCR
    def update_GCR(self,
                   mode:str = "from fgas",
                   *args,
                   **kwargs
                   ) -> float:
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
                             MLR: float = None,
                             SFR: float = None
                             ) -> float:
        """Compute (reservoir) gas change rate more 'intuitively' straight
        from overall accretion of gas into the galaxy, minus the gas that is
        expelled or turned into (long or short lived aka any kind of) stars.
        The instant return of gas from short lived stars is handled at the
        level of the integration of mstar, mgas"""
        GAR = self.GAR if GAR is None else GAR
        MLR = self.MLR if MLR is None else MLR
        SFR = self.SFR if SFR is None else SFR

        return GAR - (MLR + SFR)


    # mgas
    def update_mgas(self, timestep: float, *args, **kwargs) -> float:
        self.mgas = self.compute_mgas(timestep=timestep, *args, **kwargs)
        return self.mgas

    def compute_mgas(self,
                      timestep: float,
                      GCR: float = None,) -> float:
        GCR = self.GCR if GCR is None else GCR

        return GCR * timestep


    # mhalo
    def update_mhalo(self, timestep: float, *args, **kwargs) -> float:
        """Update the halo mass using the compute_mhalo() method"""
        self.mhalo = self.compute_mhalo(timestep, *args, **kwargs)
        return self.mhalo

    def compute_mhalo(self,
                      timestep: float,
                      HLF: float = None,
                      mhalo: float = None,
                      mout: float = None,
                      MIR: float = None,
                      ) -> float:
        """Compute new halo mass based on previous mhalo, mass increase rate
        and time step (integration), as well as precomputed (already integrated)
        baryonic mass loss from the galaxy into the halo and the fraction of
        which is also fully lost from the halo"""
        HLF = self.HLF if HLF is None else HLF
        mhalo = self.mhalo if mhalo is None else mhalo
        mout = self.mout if mout is None else mout
        MIR = self.MIR if MIR is None else MIR

        return mhalo + (MIR * timestep) - (HLF * mout)


    # MIR
    def update_MIR(self, sMIR: float = None, *args, **kwargs) -> float:
        if sMIR is None:
            self.update_sMIR()

        self.MIR = self.compute_MIR(*args, **kwargs)
        return self.MIR

    def compute_MIR(self,
                    mhalo: float = None,
                    sMIR: float = None
                    ) -> float:
        mhalo = self.mhalo if mhalo is None else mhalo
        sMIR = self.sMIR if sMIR is None else sMIR

        return sMIR * mhalo


    # MLR
    def update_MLR(self,
                   mode:str = "from fout",
                   *args,
                   **kwargs
                   ) -> float:
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
    def update_rsSFR(self,
                     mode: str = 'empirical',
                     *args,
                     **kwargs
                     ) -> float:
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