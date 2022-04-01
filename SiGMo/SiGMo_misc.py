# helper methods
import numpy as np


def GMS_Saintonge2016(logMstar):
    '''computes the log10SFR for any given log10Mstar on Galaxy Main Seqence according to Saintonge+2016, Eq.5'''
    return ((-2.332) * logMstar + 0.4156 * logMstar**2 - 0.01828 * logMstar**3)


def calc_mstar_from_mhalo(mhalo):
    """Follows Eq. 2 from Moster+2010"""
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
    """Iterate to the matching mhalo from known mstar, uses Eq.2 and Table 1 from Moster+2010"""
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
    bincentres = []
    for (value,
         x_lower, x_upper,
         y_lower, y_upper) in zip(value_arr.flat,
                                  x_mesh[:-1, :-1].flat, x_mesh[1:, 1:].flat,
                                  y_mesh[:-1, 1:].flat, y_mesh[1:, :-1].flat):
        if not np.isnan(value):
            bincentres.append([0.5 * (x_lower + x_upper), 0.5 * (y_lower + y_upper)])

    return np.array(bincentres)