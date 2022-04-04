# helper methods
import warnings

import numpy as np


def GMS_Saintonge2016(logMstar):
    """
    Computes the log10SFR for any given log10Mstar on Galaxy Main Seqence according to Saintonge+2016, Eq.5

    :param logMstar: log10(stellar mass/solar mass) of a galaxy (in units of solar masses)
    :return: log10(star-formation rate/solar mass per year) of a galaxy on the star-forming galaxy main sequence, given
    the input log10(stellar mass/solar mass)
    """
    return ((-2.332) * logMstar + 0.4156 * logMstar**2 - 0.01828 * logMstar**3)


def GMS_SFR_Speagle2014(mstar: float, z: float=None, tc: float=None, log=True):
    """
    Convenience function (thin wrapper) to use the GMS_sSFR_Speagle2014() function but return SFR instead of sSFR

    Compute the star-formation rate for galaxies on the main sequence, depending on their stellar mass and
    their redshift. Applicable redshift range: z=0-5. Applicable stellar mass range: log(mstar/M_sol)=9.0-11.8.
    Adapted from Tacconi+2020, Eq. 1, who in turn adapted it from Speagle+2014, then adapted to return SFR

    :param mstar: stellar mass of the galaxy, in units of solar mass
    :param z: redshift of the galaxy
    :param tc: cosmic time at which the galaxy exists
    :param log: flag to switch between logarithmic in/output (log=True, default), and linear in/output (log=False)
    :return: star formation rate, in units of  M_sol/Gyr
    """
    return GMS_sSFR_Speagle2014(mstar, z, tc, log) * mstar


def GMS_sSFR_Speagle2014(mstar: float, z: float=None, tc: float=None, log=True):
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
    log_sSFR = (-0.16 - 0.026 * tc) * (log_mstar + 0.025) - (6.51 - 0.11 * tc) + 9.

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


def calc_mstar_from_mhalo(mhalo):
    """
    Calculate stellar mass according to Eq. 2 from Moster+2010m with constants according to Table 1

    :param mhalo: halo mass in units of solar mass
    :return: inferred stellar mass of the galaxy in the halo, given the halo mass entered
    """
    # set constants acc. to Table 1, Moster+2010
    M_1 = 10**11.884
    m_over_M_0 = 0.02820
    beta = 1.057
    gamma = 0.556

    # calc
    mstar = mhalo * 2 * m_over_M_0
    mstar = mstar / ((mhalo/M_1)**(-beta) + (mhalo/M_1)**(gamma))

    return mstar


def iter_mhalo_from_mstar(
        mstar: float,
        precision: float=1e-3,
        i_max: int=1e5,
        initial_guess_f: float=1./0.02820,
        verbose=False
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
    mstar_derived = calc_mstar_from_mhalo(mhalo)
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
        mstar_derived = calc_mstar_from_mhalo(mhalo)
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


def calculate_mgas_mstar_from_sSFR(sSFR, log_values=False, withscatter=False):
    """
    Calculates the M_H2 / M_star ratio from the relation to SSFR according to Saintonge & Catinella 2021 (review), Eq. 5
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