# helper methods
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo

import warnings
import tqdm


# Galaxy Main Sequence WITH flattening

def GMS_Saintonge2016(logMstar):
    """
    Computes the log10SFR for any given log10Mstar on Galaxy Main Seqence according to Saintonge+2016, Eq.5

    :param logMstar: log10(stellar mass/solar mass) of a galaxy (in units of solar masses)
    :return: log10(star-formation rate/solar mass per year) of a galaxy on the star-forming galaxy main sequence, given
    the input log10(stellar mass/solar mass)
    """
    return ((-2.332) * logMstar + 0.4156 * logMstar**2 - 0.01828 * logMstar**3)


def GMS_Saintonge2022(mstar, log=True):
    """
    Computes the SFR on the Galaxy Main Sequence at z~0 according to Saintonge+Catinella2022, Eq. 7

    :param mstar: stellar mass of the galaxy in units of solar mass, either in linear values (when log=False) or in
        logarithmic values (when log=True; default)
    :param log: determines whether I/O is in lin values (log=False) in log values (default: log=True)
    :return: SFR on the GMS associated with the mstar entered, in either linear or logarithmic values dependent on log
    """
    lin_mstar = 10**mstar if log else mstar
    log_SFR = 0.412 - np.log10(1. + (lin_mstar / (10**10.59))**(-0.718))
    return log_SFR if log else 10**log_SFR


def GMS_Leslie2020(mstar, z=0., log=True):
    """
    Computes the SFR on the Galaxy Main Sequence for star-forming galaxies, according to Leslie+2020, dependent on
    stellar mass and redshift, according to Eq. 6 and Table 1 (SF)

    :param mstar: stellar mass of the galaxy in units of solar mass, either in linear values (when log=False) or in
        logarithmic values (when log=True; default)
    :param z: redshift of the galaxy
    :param log: determines whether I/O is in lin values (log=False) in log values (default: log=True)
    :return: Star Formation Rate (SFR) in units of solar mass per year on the GMS, either in lin or log values dependent
        on whether log=False or log=True (default)
    """

    # set fixed params
    S0 = 2.97
    M0 = 11.06
    a1 = 0.22
    a2 = 0.12

    # set input-based params
    t = cosmo.age(z).value
    M = mstar if log else np.log10(mstar)

    # calculation
    Mt = M0 - (a2 * t)
    log_SFR = S0 - (a1 * t) - np.log10(1. + (10**Mt / 10**M))

    # take care to return either lin or log according to log
    return log_SFR if log else 10**log_SFR


# Galaxy Main Sequence WITHOUT flattening

def GMS_SFR_Speagle2014(mstar, z: float=None, tc: float=None, log=True):
    """
    Convenience function (thin wrapper) to use the GMS_sSFR_Speagle2014() function but return SFR instead of sSFR

    Compute the star-formation rate for galaxies on the main sequence, depending on their stellar mass and
    their redshift. Applicable redshift range: z=0-5. Applicable stellar mass range: log(mstar/M_sol)=9.0-11.8.
    Adapted from Tacconi+2020, Eq. 1, who in turn adapted it from Speagle+2014, then adapted to return SFR

    :param mstar: stellar mass of the galaxy, in units of solar mass
    :param z: redshift of the galaxy
    :param tc: cosmic time at which the galaxy exists
    :param log: flag to switch between logarithmic in/output (log=True, default), and linear in/output (log=False)
    :return: star formation rate, in units of  M_sol/yr
    """
    if log:  # multiplication in lin space translates to addition in log space
        SFR = GMS_sSFR_Speagle2014(mstar, z, tc, log) + mstar
    else:
        SFR = GMS_sSFR_Speagle2014(mstar, z, tc, log) * mstar
    return SFR


def GMS_sSFR_Speagle2014(mstar, z: float=None, tc: float=None, log=True):
    """
    Compute the specific star-formation rate for galaxies on the main sequence, depending on their stellar mass and
    their redshift. Applicable redshift range: z=0-5. Applicable stellar mass range: log(mstar/M_sol)=9.0-11.8.
    Adapted from Tacconi+2020, Eq. 1, who in turn adapted it from Speagle+2014.

    :param mstar: stellar mass of the galaxy, in units of solar mass
    :param z: redshift of the galaxy
    :param tc: cosmic time at which the galaxy exists
    :param log: flag to switch between logarithmic in/output (log=True, default), and linear in/output (log=False)
    :return: specific star formation rate, in units of 1/Gyr
    """
    # switch between log and lin I/O
    log_mstar = np.log10(mstar) if not log else mstar

    # check that only exactly one of either z or tc is provided, calculate tc is necessary
    if z is None and tc is None:
        raise TypeError('Neither z (redshift) nor tc (cosmic time) were provided.\n'
                        'sSFR value for the z-dependent GMS not calculated.\n'
                        'Please provide either and run again')
    elif z is not None and tc is not None:
        raise TypeError('Both z (redshift) and tc (cosmic time) were provided.\n'
                        'sSFR value for the z-dependent GMS not calculated\n'
                        'Please provide only one and run again')
    elif z is not None:
        tc = cosmictime_Speagle2014(z, log=False)  # log=False because the lin version is used in the equations
    elif tc is not None:
        pass
    else:
        warnings.warn('Input situation of z (redshift) and tc (cosmic time) unclear.\n'
                      f'From input: z={z!r} , tc={tc!r}\n .'
                      f'Check inputs to assure calculation is sensible')

    # actual calculation
    log_sSFR = (-0.16 - 0.026 * tc) * (log_mstar + 0.025) - (6.51 - 0.11 * tc)  # without the original +9 (/Gyr --> /yr)
    # log_sSFR = (-0.16 - 0.026 * tc) * (log_mstar + 0.025) - (6.51 - 0.11 * tc) + 9.

    # take care of the right output if log=True or False
    sSFR = log_sSFR if log else 10.**log_sSFR
    return sSFR


def cosmictime_Speagle2014(z:float, log=False):
    """
    Calculate the cosmic time from the redshift, according to Tacconi+2020, used in the Speagle+2014 GMS calculations.
    It assumes a flat, ΛCDM (cold dark matter) cosmology with H_0=70 km s^−1 Mpc^−1 and Ω_m=0.3

    :param z: redshift
    :param log: flag to switch between logarithmic output (log=True), and linear output (log=False, default)
    :return: cosmic time, in units of Gyr
    """
    log_tc = (1.143 -
              1.026 * np.log10(1 + z) -
              0.599 * (np.log10(1 + z))**2 +
              0.528 * (np.log10(1 + z))**3)

    # take care of the right output if log=True or False
    tc = log_tc if log else 10.**log_tc
    return tc


# Stellar to Halo Mass Relation

def calc_mstar_from_mhalo(mhalo, z: float=0., try_lookup=True, interpolate=True):
    """
    Calculate stellar mass according to Moster+2010, with different methods/constants available

    :param mhalo: halo mass in units of solar mass
    :param z: redshift (default: 0.)
    :param try_lookup: try to use observational values from Table 6 in Moster+2010 first? (default: True)
    :param interpolate: interpolate either for values not in table or always, depending on try_lookup? (default: True)
    :return: inferred stellar mass of the galaxy in the halo, given the halo mass entered
    """
    # set constants acc. to Table 1, Moster+2010
    log_M_1, m_over_M_0, beta, gamma = return_stellar_to_halo_mass_ratio_parameters_for_z(z=z,
                                                                                          try_lookup=try_lookup,
                                                                                          interpolate=interpolate)
    M_1 = 10**log_M_1

    # calc
    mstar = mhalo * 2 * m_over_M_0
    mstar = mstar / ((mhalo/M_1)**(-beta) + (mhalo/M_1)**(gamma))

    return mstar


def lookup_stellar_to_halo_mass_ratio_parameters_for_z(z:float=0.):
    """
    Looks up the parameter values for the Moster+2010 stellar-to-halo mass relation depending on redshift. If not
    exactly a redshift as specified in Table 6 of Moster+2010, a Warning will be raised. For values not covered
    exactly by this table, please refer to the interpolation function from Moster+2010, Eqns. 23...26 and Table 7

    :param z: redshift for the relation (default: 0.)
    :return: parameters log_M_1, m_over_M_0, beta and gamma
    """
    # make the table with values from the Moster+2010 paper
    param_table = Table()
    param_table['z'] = [0.0, 0.5, 0.7, 0.9, 1.1, 1.5, 1.8, 2.5, 3.5]
    param_table['log_M_1'] = [11.88, 11.95, 11.93, 11.98, 12.05, 12.15, 12.28, 12.22, 12.21]
    param_table['m_over_M_0'] = [0.0282, 0.0254, 0.0215, 0.0142, 0.0175, 0.0110, 0.0116, 0.0130, 0.0101]
    param_table['beta'] = [1.057, 1.37, 1.18, 0.91, 1.66, 1.29, 1.53, 0.90, 0.82]
    param_table['gamma'] = [0.556, 0.55, 0.48, 0.43, 0.52, 0.41, 0.41, 0.30, 0.46]

    param_selected = param_table[param_table['z'] == z]
    if len(param_selected) == 0:
        warnings.warn("Warning: the entered redshift (z) does not correspond to an entry in Table 6 in Moster+2010!\n"
                      "No parameters will be returned!")
        param_table['z'] = [None]
        param_table['log_M_1'] = [None]
        param_table['m_over_M_0'] = [None]
        param_table['beta'] = [None]
        param_table['gamma'] = [None]

    param_selected.remove_column('z')
    return param_selected[0]


def interpolate_stellar_to_halo_mass_ratio_parameters_for_z(z:float=0.):
    """
    Interpolate the parameters for the Moster+2010 stellar-to-halo mass relation according to Eqns. 23...26 and Table 7

    :param z: redshift (default: 0.)
    :return: parameters log_M_1, m_over_M_0, beta and gamma
    """

    # set mu and nu
    mu = 0.019
    nu = -0.72
    gamma1 = -0.26
    beta1 = 0.17

    log_M_1_z0, m_over_M_0_z0, beta_z0, gamma_z0 = lookup_stellar_to_halo_mass_ratio_parameters_for_z(z=0.)
    log_M_1 = log_M_1_z0 * (z + 1)**mu
    m_over_M_0 = m_over_M_0_z0 * (z + 1)**nu
    beta = beta_z0 + beta1 * z
    gamma = gamma_z0 * (z + 1)**gamma1

    return log_M_1, m_over_M_0, beta, gamma


def return_stellar_to_halo_mass_ratio_parameters_for_z(z: float=0., try_lookup=True, interpolate=True):
    """
    Returns the parameters for the Moster+2010 stellar-to-halo mass relation from different methods

    :param z: redshift (default: 0.)
    :param try_lookup: try to use observational values from Table 6 in Moster+2010 first? (default: True)
    :param interpolate: interpolate either for values not in table or always, depending on try_lookup? (default: True)
    :return: parameters log_M_1, m_over_M_0, beta and gamma
    """
    if try_lookup:  # try obs. fitted values first
        params = lookup_stellar_to_halo_mass_ratio_parameters_for_z(z)
        if params[0] is None:  # if z is not in the table
            if interpolate:  # interpolate in that case
                params = interpolate_stellar_to_halo_mass_ratio_parameters_for_z(z)
            else:  # only want to use table values
                raise ValueError(f"Redshift z={z} is not in Table 6 of Moster+2010, and interpolate is set to False.\n"
                                 f"Use appropriate redshift or enable interpolation as per Moster+2010")
    else:  # only use fit ("interpolate")
        params = interpolate_stellar_to_halo_mass_ratio_parameters_for_z(z)

    return params


def iter_mhalo_from_mstar(
        mstar: float,
        precision: float=1e-3,
        i_max: int=1e5,
        initial_guess_f: float=1./0.02820,
        verbose=False,
        z: float=0.,
        try_lookup=True,
        interpolate=True
) -> float:
    """
    Iterate to the matching mhalo from known mstar, using the known relation from mstar to mhalo and narrowing in
    towards the initially entered mstar value. Uses `calc_mstar_from_mhalo` and thus Eq.2 and Table 1 from Moster+2010

    :param mstar: stellar mass of the galaxy in units of solar mass
    :param precision: relative error of how close to the true value of mstar does the inferred mstar value from the
    mhalo-to-mstar relation need to be? (default: 1e-3)
    :param i_max: maximum number of iterations (default: 1e5)
    :param initial_guess_f: initial guess of factor between mhalo and mstar. The default value is the inverse of the
    (m/M)_0 value from Table 1 in Moster+2010, which is the leading factor (default: 1./0.02820)
    :param verbose: will details about results and number of iterations be output to the terminal? (default: False)
    :return: mhalo: the halo mass inferred from the stellar mass of the galaxy, given the relation from Moster+2010
    """
    # inital mhalo guess and derived mstar
    mhalo = initial_guess_f*mstar
    mstar_derived = calc_mstar_from_mhalo(mhalo, z=z, try_lookup=try_lookup, interpolate=interpolate)
    ratio = mstar_derived/ mstar
    deviation = ratio - 1

    # define intial shrink factor
    shrink_f = 0.9

    # loop towards relation
    i = 0
    while abs(deviation) > precision:
        # back-up prev values
        mhalo_prev = mhalo
        deviation_prev = deviation

        # adjust mhalo
        mhalo = mhalo - shrink_f * (mhalo * deviation)

        # derive new mstar_derived and calc. ratio and deviation
        mstar_derived = calc_mstar_from_mhalo(mhalo, z=z, try_lookup=try_lookup, interpolate=interpolate)
        ratio = mstar_derived/ mstar
        deviation = ratio - 1

        if type(deviation) == complex or abs(deviation) > abs(deviation_prev):
            # if type(deviation) == complex or deviation > deviation_prev:
            shrink_f *= 0.5
            mhalo = mhalo_prev
            deviation = deviation_prev

        # increase i
        i += 1

        # emergency exit condition
        if i >= i_max:
            print(f"WARNING: did NOT converge after {i} iterations! ", "\n",
                  f"Preliminary mhalo = {mhalo:.3e} from mstar = {mstar:.3e}.", "\n",
                  f"Remaining deviation = {deviation:.3e}, did not reach precision = {precision:.3e}")
            break

    if verbose:
        print(f"Computed mhalo = {mhalo:.3e} from mstar = {mstar:.3e}\n"
              f"Precision = {precision:.3e}, iterations = {i}")
    return mhalo


# Gas Mass to Stellar Mass Relation

def calculate_mgas_mstar_from_sSFR_Saintonge2022(sSFR, log_values=False, withscatter=False):
    """
    Calculates the M_H2 / M_star ratio from the relation to SSFR according to Saintonge & Catinella 2022 (review), Eq. 5
    Input values can be either in log or lin, and scatter can be included in the output as well.

    :param sSFR: Specific star formation rate in solar masses per year, and in log(M_sol/yr) if log_values = True
    :param log_values: Whether the input and output values are in log space or not (default: False)
    :param withscatter: Whether the intrins. and obs. scatter are computed and included in the output (Default: False)
    :return: The ratio of M_H2 (molecular gas mass) and M_star (stellar mass) in solar masses per year, and in
    log(M_sol/yr) if log_values = True
    """
    sSFR = np.array(sSFR)
    if not log_values:
        sSFR = np.log10(sSFR)
    if not withscatter:
        t1 = np.array([0.,])
        t2 = np.array([0.,])
    else:
        t1 = np.array([-0.01, 0., +0.01])
        t2 = np.array([-0.12, 0., +0.12])

    res = []
    for _sSFR in sSFR:
        _tmp = (0.75 + t1) * _sSFR + (6.24 + t2)
        if not log_values:
            _tmp = 10**_tmp
        _tmp = tuple(_tmp) if len(tuple(_tmp)) > 1 else _tmp[0]
        res.append(_tmp)

    if len(res) == 1:
        res = res[0]
    return res


def calculate_mgas_mstar_from_sSFR_Tacconi2018(sSFR, mstar, z, fit=True):
    # fit parameters acc. to Table 3(b), Tacconi+2018: "Best β=2 W14"
    beta = 2

    A = 0.16
    dA = 0.15

    B = -3.69
    dB = 0.4

    F = 0.65
    dF = 0.1

    C = 0.52
    dC = 0.03

    D = -0.36
    dD = 0.03

    # need to hand-in
    deltaMstar = 1

    # preliminary calculations

    # THIS DOES NOT WORK (YET)!!!
    sSFR_MS = None  # sSFR from W14 main-sequence (NOT IMPLEMENTED "YET", and we switched to other prescription)
    # THIS DOES NOT WORK (YET)!!!

    deltaMstar = mstar / (5 * 10**10)

    # calculate the fit
    deltaMS = sSFR / sSFR_MS
    log_mu = A + B * (np.log10(1 + z) - F)**beta + C * np.log10(deltaMS) + D * np.log10(deltaMstar)


    # this is the general form, not the fit (as the latter requires information about the half-light radius at 5000 Å)

    # not done "YET" (maybe never, due to switch to Tacconi+2020 prescription


    return


def calculate_mgas_mstar_from_sSFR_Tacconi2020(sSFR, mstar, z, log=True, withscatter=False):
    sSFR = np.array(sSFR)
    mstar = np.array(mstar)

    # set fit parameters
    A = np.array([0.06]*3) + [-0.2, 0., 0.2]
    B = np.array([-3.33]*3) + [-0.2, 0., 0.2]
    F = np.array([0.65]*3) + [-0.05, 0., 0.05]
    C = np.array([0.51]*3) + [-0.03, 0., 0.03]
    D = np.array([-0.41]*3) + [-0.03, 0., 0.03]

    # calculate for each entry
    log_res = []
    for _sSFR, _mstar in zip(sSFR, mstar):
        # calculate Speagle+14 MS sSFR
        sSFR_MS = GMS_sSFR_Speagle2014(mstar=_mstar, z=z, log=log)

        # compute individ. terms (depending on whether values are handled linearly of logarithmically)
        Term_AB = A + B * (np.log10(1 + z) - F)**2
        Term_C = C * np.log10(_sSFR/sSFR_MS) if not log else C * (_sSFR - sSFR_MS)
        Term_D = D * (np.log10(_mstar) - 10.7) if not log else D * (_mstar - 10.7)

        # add terms together
        log_mgas_mstar = Term_AB + Term_C + Term_D

        # do we return only the value or value and scatter?
        log_res.append(log_mgas_mstar if withscatter else log_mgas_mstar[1])
    log_res = np.array(log_res)

    return log_res if log else 10**log_res


# Useful Helper Methods

def calc_bincentres_where_not_nan(value_arr, x_mesh, y_mesh):
    """
    Calculates the bin centres from a 2-d array of values (value_arr) und two 2-d mesh arrays with corresponding bin
    edges, for all 2-d bins where value_arr is not np.nan. Example: value_arr.shape = (10, 10) , then x_mesh.shape =
    (11, 11) and y_mesh.shape = (11, 11) , as produced by np.meshgrid().

    :param value_arr: 2-d array with values, e.g. bin counts. Non-relevant bins should be np.nan, so that their bin
    centres are not calculated here and not included in the return array
    :param x_mesh: 2-d array as produced by np.meshgrid() based on the bin edges of value_arr in x direction
    :param y_mesh: 2-d array as produced by np.meshgrid() based on the bin edges of value_arr in y direction
    :return: 2-d array with shape=(n,2), n being the entries in value_arr that are not np.nan, and the second dimension
    having the x and y coordinates of the bin centre
    """
    bincentres = []
    for (value,
         x_lower, x_upper,
         y_lower, y_upper) in zip(value_arr.flat,
                                  x_mesh[:-1, :-1].flat, x_mesh[1:, 1:].flat,
                                  y_mesh[:-1, 1:].flat, y_mesh[1:, :-1].flat):
        if not np.isnan(value):
            bincentres.append([0.5 * (x_lower + x_upper), 0.5 * (y_lower + y_upper)])

    return np.array(bincentres)


def find_nearest_index(arr, val):
    """
    Determines which index in arr (array) contains the value closest to val (value). If arr contains several elements of
    the same value who all would fulfil that condition equally, the first element in order of traversion is returned.

    :param arr: an array in which to search for the nearest match
    :param val: the value for which the closest match is searched
    :return: index in arr pointing to the (first) closest element to val
    """
    arr = np.asarray(arr)
    nearest_i = (np.abs(arr - val)).argmin()
    return nearest_i


# SFR79 Star Formation Rate Change Parameter

def compute_SFR79_from_SFR(gal_a):
    """
    Helper function to compute the star-formation change parameter SFR79 directly
    from the SFR over time data

    :param gal_a: numpy array of SiGMo.Snapshot objects over time. Can have arbitrary
    shape, as long as the different Snapshots over time for each object are in the
    last dimension
    :return: SFR79_a: numpy array with same shape as gal_a except for the last
    dimension; the time dimension is replaced by three summary values returned
    in the new last dimension:
     [0] average SFR over 5 Myr (in units of M_sol);
     [1] average SFR over 800 Myr (in units of M_sol);
     [2] log SFR 79 (being log10(avrgSFR_5Myr / avrgSFR_800Myr).
    Example: if gal_a.shape = (10, 16, 4, 1999), then SFR79.shape = (10, 16, 4, 3)
    """
    SFR79_a = np.empty(shape=(*gal_a.shape[:-1], 3), dtype=object)
    SFR79_a_fl = SFR79_a.reshape(-1, *SFR79_a.shape[-1:])
    gal_a_fl = gal_a.reshape(-1, *gal_a.shape[-1:])
    for i, gal_seq in enumerate(tqdm(gal_a_fl)):
        lookbacktime_a = np.array([gal.data["lookbacktime"] for gal in gal_seq])
        SFR_a = np.array([gal.data["SFR"] for gal in gal_seq])

        # avrg over 5 Myr
        SFR7_a = SFR_a[lookbacktime_a <= 0.005]
        SFR7_avg = np.sum(SFR7_a) / len(SFR7_a)
        SFR79_a_fl[i, 0] = SFR7_avg
        # print(SFR79_grid_fl[i, 0])

        # avrg over 800 Myr
        SFR9_a = SFR_a[lookbacktime_a <= 0.8]
        SFR9_avg = np.sum(SFR9_a) / len(SFR9_a)
        SFR79_a_fl[i, 1] = SFR9_avg
        # print(SFR79_grid_fl[i, 1])

        # log10(SFR_5Myr / SFR_800Myr)
        SFR79_a_fl[i, 2] = np.log10(SFR7_avg / SFR9_avg)
        # print(SFR79_grid_fl[i, 2])

    return SFR79_a


def join_paths(path1, path2):
    """
    Join two paths or a path and a filename, assuming that if they're not strings, then they're path objects

    :param path1: first path
    :param path2: second path
    :return: combined paths
    """
    if (type(path1) is str) and (type(path2) is str):
        if (path1[-1] == "/") ^ (path2[0] == "/"):
            combipath = path1 + path2
        elif (path1[-1] == "/") and (path2[0] == "/"):
            combipath = path1[:-1] + path2
        else:
            combipath = path1 + "/" + path2
    else:
        combipath = path1 / path2
    return combipath


def assign_with_warning(target, value, warning: bool = True):
    """
    Simple assigning of value to target that warns if value not 'equal' ot target

    :param target: target variable whose value will be overwritten by value
    :param value: new value to be written into target
    :param warning: flags whether non-equality between target and value raises an error (default: True)
    :return: always returns value
    """
    if (value != target) and warning:
        warnings.warn(f"WARNING! {target!r} and {value!r} not identical!")
    return value