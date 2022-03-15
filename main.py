import numpy

import SiGMo as sgm

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

from tqdm import tqdm, trange
from timeit import default_timer as timer
from datetime import timedelta, datetime

# ====================
# set some global vars
alphabet_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# =====================
# some helper functions
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

def plot_initial_conditions(mstar, SFR, mgas, mhalo, mstar_mesh, sfr_mesh, sfr79_medians, n_binned_min, plot_dir):
    sfr79_range = (-2, 2)
    cmap = mpl.cm.RdBu
    norm = mpl.colors.Normalize(vmin=sfr79_range[0], vmax=sfr79_range[1])
    fig, (ax_obs, ax_cbar, ax_cbar_mhalo, ax_cbar_mgas) = plt.subplots(1, 4,
                                                                       gridspec_kw={
                                                                           'width_ratios': (9, 1, 1, 1),
                                                                           'hspace': 0.05
                                                                       },
                                                                       figsize=(15, 9))
    im_obs = ax_obs.pcolormesh(mstar_mesh, sfr_mesh,
                               sfr79_medians,
                               cmap=cmap, norm=norm)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax_cbar,
                 fraction=0.8,
                 extend='both',
                 anchor=(0.0, 0.0),
                 label='log SFR79')
    # adding the Galaxy Main Sequence on top (Saintonge+2016, Eq. 5)
    GMS_x = np.linspace(start=np.min(mstar_mesh),
                        stop=np.max(mstar_mesh),
                        num=1000,
                        endpoint=True)
    ax_obs.plot(GMS_x, GMS_Saintonge2016(GMS_x), color='xkcd:magenta', ls='--')
    mhalo_color_range = (np.min(np.log10(mhalo)), np.max(np.log10(mhalo)))
    cmap_mhalo = mpl.cm.RdPu
    norm_mhalo = mpl.colors.Normalize(vmin=mhalo_color_range[0], vmax=mhalo_color_range[1])
    ic_sims_mhalo = ax_obs.scatter(x=np.log10(mstar), y=np.log10(SFR) - 9.,
                                   c=np.log10(mhalo),
                                   facecolors='none', edgecolors='face',
                                   cmap=cmap_mhalo, norm=norm_mhalo,
                                   marker="o", s=1.5 * mpl.rcParams['lines.markersize'] ** 2.,
                                   zorder=10)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm_mhalo, cmap=cmap_mhalo),
                 ax=ax_cbar_mhalo,
                 fraction=0.8,
                 # extend='both',
                 anchor=(0.0, 0.0),
                 label=r'log $M_{\mathrm{Halo}}$ [$M_\odot$]')
    mgas_color_range = (np.min(np.log10(mgas)), np.max(np.log10(mgas)))
    cmap_mgas = mpl.cm.YlGn
    norm_mgas = mpl.colors.Normalize(vmin=mgas_color_range[0], vmax=mgas_color_range[1])
    ic_sims_mgas = ax_obs.scatter(x=np.log10(mstar), y=np.log10(SFR) - 9.,
                                  c=np.log10(mgas), cmap=cmap_mgas, norm=norm_mgas,
                                  marker="o", s=1.5 * (mpl.rcParams['lines.markersize'] / 3.) ** 2.,
                                  zorder=15)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm_mgas, cmap=cmap_mgas),
                 ax=ax_cbar_mgas,
                 fraction=0.8,
                 # extend='both',
                 anchor=(0.0, 0.0),
                 label=r'log $M_{\mathrm{H}_2}$ [$M_\odot$]')
    ax_cbar.remove()
    ax_cbar_mgas.remove()
    ax_cbar_mhalo.remove()
    ax_obs.set_xlabel(r'log $M_\star$ [$M_\odot$]')
    ax_obs.set_ylabel(r'log SFR [$M_\odot \, yr^{-1}$]')
    ax_obs.text(0.05, 0.95, f"only showing bins with {n_binned_min} or more objects", transform=ax_obs.transAxes)
    fig.savefig(plot_dir / f'mstar_SFR_obs_vs_ICs_of_sims_{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}.png')
    return fig, ax_obs

def explore_attrs_and_augment_plot(env, mstar_mesh, sfr_mesh, fig, ax_obs, plot_dir):
    # min and max of valid mstar and SFR input
    mstar_min, mstar_max = np.min(mstar_mesh), np.max(mstar_mesh)
    sfr_min, sfr_max = np.min(sfr_mesh), np.max(sfr_mesh)
    # input loop: only exits once valid mstar and SFR values have been entered
    while True:
        mstar_in = input("Enter: log mstar [M_sol]")
        sfr_in = input("Enter: log SFR [M_sol/yr]")
        try:
            mstar_in = float(mstar_in)
            sfr_in = float(sfr_in)
        except ValueError:
            print("Either mstar or SFR or both were not valid floating point numbers. Please try again")
            continue
        if (mstar_min <= mstar_in <= mstar_max) and (sfr_min <= sfr_in <= sfr_max):
            break
        else:
            print(f"Please enter a valid range of mstar and SFR:\n"
                  f"{mstar_min} ≤ log mstar ≤ {mstar_max}\n"
                  f"{sfr_min} ≤ log SFR ≤ {sfr_max}")
    # find the closest value in log mstar and log SFR space that is actually used in (an) AstroObject(s)
    mstar_in_nn = min((mstar_mesh[0, :-1] + mstar_mesh[0, 1:]) / 2., key=lambda x: abs(x - mstar_in))
    sfr_in_nn = min((sfr_mesh[:-1, 0] + sfr_mesh[1:, 0]) / 2., key=lambda x: abs(x - sfr_in))
    print(f"INPUT:\n log mstar = {mstar_in}\n log SFR = {sfr_in}")
    print(f"NEAREST MATCH:\n log mstar = {mstar_in_nn}\n log SFR = {sfr_in_nn}")
    # mark the selected mstar SFR bin in the plot
    mstar_line = ax_obs.axvline(mstar_in_nn, color="xkcd:dark grey")
    sfr_line = ax_obs.axhline(sfr_in_nn, color="xkcd:dark grey")
    fig.show()
    # find the Galaxy that matches this value pair, and the Halo that contains it
    # grab all mstar and SFR values (works for ONE galaxy per halo)
    # (if I wanted to change that: env --> galaxies --> halo should be unique and work for arb. gals per halo)
    mstar_all = np.log10(np.array([_halo.galaxies[0].mstar for _halo in env.halos]))
    sfr_all = np.log10(np.array([_halo.galaxies[0].SFR for _halo in env.halos])) - 9.  # convert from Gyr to yr
    # print(sfr_all)
    # print(len(mstar_all), len(sfr_all))
    # find the matches in mstar, then in SFR, then combine
    mstar_match = np.isclose(mstar_all, mstar_in_nn)
    sfr_match = np.isclose(sfr_all, sfr_in_nn)
    mstar_sfr_match = mstar_match & sfr_match
    # check whether the selcted coordinates actually contain a galaxy and halo
    try:
        i_match = int(np.argwhere(mstar_sfr_match))
    except TypeError:
        print("The mstar-SFR bin you selected does not contain a Galaxy/Halo.\n"
              "Please restart and try again")
    else:
        halo_match = env.halos[i_match]
        gal_match = halo_match.galaxies[0]  # (this also only works for one Galaxy per Halo)

        # printing the matched AstroObject attributes
        print(f"Printing attributes of selected Galaxy and associated Halo\n"
              f"with log mstar = {mstar_all[mstar_sfr_match]}\n"
              f"and log SFR = {sfr_all[mstar_sfr_match]}")
        halo_snap = halo_match.make_snapshot()
        gal_snap = gal_match.make_snapshot()
        print("HALO PROPERTIES:")
        for k, v in halo_snap.data.items():
            print(f"  {k:<5} = {v}")
        print("GALAXY PROPERTIES")
        for k, v in gal_snap.data.items():
            print(f"  {k:<5} = {v}")

        # extend figure and add one axes at the bottom to display selected Halo's and Galaxy's properties
        fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] * 3 / 2)
        fig.subplots_adjust(bottom=0.35)
        ax_values = fig.add_axes(rect=[0.1, 0.05, 0.8, 0.25])
        ax_values.set_axis_off()

        # print halo properties to axes
        ax_values.text(0.05, 0.95, "HALO PROPERTIES:",
                       weight='bold', fontsize=16,
                       va='top',
                       transform=ax_values.transAxes)
        halo_d = halo_snap.data
        halo_d = {k: v for k, v in halo_d.items() if isinstance(v, (float, int))}
        halo_values_str = "\n".join([f"{k:<5} = {v:.3e}" for k, v in halo_d.items()])
        # halo_values_str = (f"{'mtot':<5} = {halo_d['mtot']:.3e}\n"
        #                    f"{'mdm':<5} = {halo_d['mdm']:.3e}\n"
        #                    f"{'mgas':<5} = {halo_d['mgas']:.3e}")
        ax_values.text(0.05, 0.83, halo_values_str,
                       fontsize=14, linespacing=1.3,
                       va='top',
                       transform=ax_values.transAxes)

        # print galaxy properties to axes
        ax_values.text(0.35, 0.95, "GALAXY PROPERTIES:",
                       weight='bold', fontsize=16,
                       va='top',
                       transform=ax_values.transAxes)
        gal_d = gal_snap.data
        gal_d = {k: v for k, v in gal_d.items() if isinstance(v, (float, int))}
        gal_d_items = list(gal_d.items())
        gal_values_str_1 = "\n".join([f"{k:<5} = {v:.3e}" for k, v in gal_d_items[:12]])
        ax_values.text(0.35, 0.83, gal_values_str_1,
                       fontsize=14, linespacing=1.3,
                       va='top',
                       transform=ax_values.transAxes)
        gal_values_str_2 = "\n".join([f"{k:<5} = {v:.3e}" for k, v in gal_d_items[12:]])
        ax_values.text(0.65, 0.83, gal_values_str_2,
                       fontsize=14, linespacing=1.3,
                       va='top',
                       transform=ax_values.transAxes)

        # add figure title and show figure
        title_str = f"Properties of {halo_snap.data['name']} / {gal_snap.data['name']}"
        subtitle_str = (r"$\log (M_\mathrm{halo} / \; M_\odot)$ = " + f"{np.log10(halo_snap.data['mtot']):.2f} , "
                                                                      r"$\log (M_\star / \; M_\odot)$ = " + f"{np.log10(gal_snap.data['mstar']):.2f} , "
                                                                                                            r"$\log (SFR / \; M_\odot yr^{-1})$ = " + f"{np.log10(gal_snap.data['SFR']) - 9.:.2f}")  # convert /Gyr to /yr

        fig.suptitle(title_str, fontsize=24, va='top')
        ax_values.text(0.5, 3.6, subtitle_str,
                       va='top', ha='center', fontsize=18,
                       transform=ax_values.transAxes)
        fig.show()
        fig.savefig(
            plot_dir / f'mstar_SFR_obs_vs_ICs_of_sims_details_{halo_snap.data["name"]}_{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}.png')


# ================
# various testruns


def testrun_1(timestep: float = 1.e-3, silent: bool = False):
    EnvA = sgm.Environment(name='Environment A')
    if not silent:
        print(f"Created '{EnvA.name}'")
    GalA1 = EnvA.create_Galaxy(name='Galaxy A1', mgas=1e10, mhalo=10**11.5)
    if not silent:
        print(f"Created '{GalA1.name}'")
        print(EnvA)
        print(GalA1)

    if not silent:
        print("Running a simulation...")
    GalA1_history = []
    GalA1_history.append(GalA1.make_snapshot())
    if not silent:
        print(f"{0}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")
    for t in range(10000):
        EnvA.reference_evolve(timestep=timestep)
        GalA1_history.append(GalA1.make_snapshot())
        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    if not silent:
        print(EnvA)
        print(GalA1)


    # timesteps list for plotting
    timesteps_l = np.linspace(0, len(GalA1_history) * timestep, len(GalA1_history))

    # grad the data we want to plot from history of galaxy
    mgas_l = [GalA1_history[i]['mgas'] for i, item in enumerate(GalA1_history)]
    mstar_l = [GalA1_history[i]['mstar'] for i, item in enumerate(GalA1_history)]
    mout_l = [GalA1_history[i]['mout'] for i, item in enumerate(GalA1_history)]
    GAR_l = [GalA1_history[i]['GAR'] for i, item in enumerate(GalA1_history)]
    mhalo_l = [GalA1_history[i]['mhalo'] for i, item in enumerate(GalA1_history)]
    sMIR_l = [GalA1_history[i]['sMIR'] for i, item in enumerate(GalA1_history)]
    rsSFR_l = [GalA1_history[i]['rsSFR'] for i, item in enumerate(GalA1_history)]
    fout_l = [GalA1_history[i]['fout'] for i, item in enumerate(GalA1_history)]
    fgas_l = [GalA1_history[i]['fgas'] for i, item in enumerate(GalA1_history)]
    fstar_l = [GalA1_history[i]['fstar'] for i, item in enumerate(GalA1_history)]

    fig, ax = plt.subplots(figsize=(9,6))
    # ax.plot(timesteps_l, mgas_l, label="mgas")
    # ax.plot(timesteps_l, mstar_l, label="mstar")
    # ax.plot(timesteps_l, mout_l, label="mout")
    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l, mhalo_l, label="mhalo (total halo mass)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")
    # ax.set_yscale('log')
    ax.legend()
    fig.show()

    return


def testrun_2(timestep: float = 1.e-3, silent: bool = False):
    EnvA = sgm.Environment(name='Environment A')
    if not silent:
        print(f"Created '{EnvA.name}'")
    GalA1 = EnvA.create_Galaxy(name='Galaxy A1', mgas=1e10, mhalo=10**11.5)
    if not silent:
        print(f"Created '{GalA1.name}'")
        print(EnvA)
        print(GalA1)

    if not silent:
        print("Running a simulation...")
    GalA1_history = []
    GalA1_history.append(GalA1.make_snapshot())
    if not silent:
        print(f"{0}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")
    for t in range(10000):
        EnvA.intuitive_evolve(timestep=timestep)
        GalA1_history.append(GalA1.make_snapshot())
        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    if not silent:
        print(EnvA)
        print(GalA1)


    # timesteps list for plotting
    timesteps_l = np.linspace(0, len(GalA1_history) * timestep, len(GalA1_history))

    # grad the data we want to plot from history of galaxy
    mgas_l = [GalA1_history[i]['mgas'] for i, item in enumerate(GalA1_history)]
    mstar_l = [GalA1_history[i]['mstar'] for i, item in enumerate(GalA1_history)]
    mout_l = [GalA1_history[i]['mout'] for i, item in enumerate(GalA1_history)]
    GAR_l = [GalA1_history[i]['GAR'] for i, item in enumerate(GalA1_history)]
    mhalo_l = [GalA1_history[i]['mhalo'] for i, item in enumerate(GalA1_history)]
    sMIR_l = [GalA1_history[i]['sMIR'] for i, item in enumerate(GalA1_history)]
    rsSFR_l = [GalA1_history[i]['rsSFR'] for i, item in enumerate(GalA1_history)]
    fout_l = [GalA1_history[i]['fout'] for i, item in enumerate(GalA1_history)]
    fgas_l = [GalA1_history[i]['fgas'] for i, item in enumerate(GalA1_history)]
    fstar_l = [GalA1_history[i]['fstar'] for i, item in enumerate(GalA1_history)]

    fig, ax = plt.subplots(figsize=(9,6))
    # ax.plot(timesteps_l[5:], mgas_l[5:], label="mgas")
    # ax.plot(timesteps_l[5:], mstar_l[5:], label="mstar")
    # ax.plot(timesteps_l[5:], mout_l[5:], label="mout")
    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l, mhalo_l, label="mhalo (total halo mass)")
    ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")
    ax.set_yscale('log')
    ax.legend()
    fig.show()

    return


def testrun_3(with_burnin = True, timestep: float = 1.e-3, silent: bool = False):
    EnvA = sgm.Environment(name='Environment A', zstart=6.)
    if not silent:
        print(f"Created '{EnvA.name}'")
    GalA1 = EnvA.create_Galaxy(with_burnin=True,
                               with_burnin_dict={'cycle_steps': 50,
                                                 'fixed_attr': ['mhalo'],
                                                 'check_attr': ['mgas', 'mstar', 'SFR', 'GCR', 'MLR']
                                                 },
                               name='Galaxy A1',
                               mgas=1e10,
                               mhalo=10**11.5
                               )
    if not silent:
        print(f"Created '{GalA1.name}'")
        print(EnvA)
        print(GalA1)

    if not silent:
        print("Running a simulation...")
    GalA1_history = []
    GalA1_history.append(GalA1.make_snapshot())
    if not silent:
        print(f"{0}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")
    for t in range(10000):
        EnvA.intuitive_evolve(timestep=timestep)  # combine this line and the next?
        GalA1_history.append(GalA1.make_snapshot())
        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    if not silent:
        print(EnvA)
        print(GalA1)


    # timesteps list for plotting
    timesteps_l = np.linspace(0, len(GalA1_history) * timestep, len(GalA1_history))

    # grab the data we want to plot from history of galaxy
    z_l = [GalA1_history[i]['z'] for i, item in enumerate(GalA1_history)]

    mgas_l = [GalA1_history[i]['mgas'] for i, item in enumerate(GalA1_history)]
    mstar_l = [GalA1_history[i]['mstar'] for i, item in enumerate(GalA1_history)]
    mout_l = [GalA1_history[i]['mout'] for i, item in enumerate(GalA1_history)]
    mhalo_l = [GalA1_history[i]['mhalo'] for i, item in enumerate(GalA1_history)]

    # GAR_l = [GalA1_history[i]['GAR'] for i, item in enumerate(GalA1_history)]
    # sMIR_l = [GalA1_history[i]['sMIR'] for i, item in enumerate(GalA1_history)]
    # rsSFR_l = [GalA1_history[i]['rsSFR'] for i, item in enumerate(GalA1_history)]

    GCR_l = [GalA1_history[i]['GCR'] for i, item in enumerate(GalA1_history)]
    SFR_l = [GalA1_history[i]['SFR'] for i, item in enumerate(GalA1_history)]
    MLR_l = [GalA1_history[i]['MLR'] for i, item in enumerate(GalA1_history)]
    MIR_l = [GalA1_history[i]['MIR'] for i, item in enumerate(GalA1_history)]

    # fout_l = [GalA1_history[i]['fout'] for i, item in enumerate(GalA1_history)]
    # fgas_l = [GalA1_history[i]['fgas'] for i, item in enumerate(GalA1_history)]
    # fstar_l = [GalA1_history[i]['fstar'] for i, item in enumerate(GalA1_history)]

    # ========
    # plotting

    fig_masses, ax_masses = plt.subplots(figsize=(9,6))
    ax_masses.plot(z_l[5:], mhalo_l[5:], label="mhalo (halo mass)", c="xkcd:red")
    ax_masses.plot(z_l[5:], mgas_l[5:], label="mgas (gas mass)", c="xkcd:blue")
    ax_masses.plot(z_l[5:], mstar_l[5:], label="mstar (stellar mass)", c="xkcd:orange")
    ax_masses.plot(z_l[5:], mout_l[5:], label="mout (ejected masss)", c="xkcd:green")
    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Redshift')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')
    ax_masses.legend()
    fig_masses.show()


    fig_rates, ax_rates = plt.subplots(figsize=(9,6))
    ax_rates.plot(z_l[5:], MIR_l[5:], label="MIR (mass increase)", c="xkcd:red")
    ax_rates.plot(z_l[5:], GCR_l[5:], label="GCR (gas change)", c="xkcd:blue")
    ax_rates.plot(z_l[5:], SFR_l[5:], label="SFR (star formation)", c="xkcd:orange")
    ax_rates.plot(z_l[5:], MLR_l[5:], label="MLR (mass loss)", c="xkcd:green")

    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Redshift')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')
    ax_rates.legend()
    fig_rates.show()

    return


def testrun_4(with_burnin = True, timestep: float = 1.e-3, silent: bool = False):
    EnvA = sgm.Environment(name='Environment A', zstart=6.)
    if not silent:
        print(f"Created '{EnvA.name}'")
    HaloA1 = EnvA.create_Halo(name='Halo A1')
    GalA1 = HaloA1.create_Galaxy(with_burnin=True,
                                 with_burnin_dict={'cycle_steps': 5,
                                                   'fixed_attr': ['mhalo', 'mstar'],
                                                   'check_attr': ['mgas', 'SFR', 'GCR', 'MLR']
                                                   },
                                 name='Galaxy A1',
                                 mgas=1e10,
                                 mhalo=10**11.5
                                 )
    if not silent:
        print(f"Created '{GalA1.name}'")
        print(EnvA)
        print(GalA1)

    if not silent:
        print("Running a simulation...")
    GalA1_history = []
    GalA1_history.append(GalA1.make_snapshot())
    if not silent:
        print(f"{0}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")
    for t in range(10000):
        EnvA.evolve(timestep=timestep, mode='intuitive')  # combine this line and the next?
        GalA1_history.append(GalA1.make_snapshot())
        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    if not silent:
        print(EnvA)
        print(GalA1)


    # timesteps list for plotting
    timesteps_l = np.linspace(0, len(GalA1_history) * timestep, len(GalA1_history))

    # grab the data we want to plot from history of galaxy
    z_l = [GalA1_history[i]['z'] for i, item in enumerate(GalA1_history)]

    mgas_l = [GalA1_history[i]['mgas'] for i, item in enumerate(GalA1_history)]
    mstar_l = [GalA1_history[i]['mstar'] for i, item in enumerate(GalA1_history)]
    mout_l = [GalA1_history[i]['mout'] for i, item in enumerate(GalA1_history)]
    mhalo_l = [GalA1_history[i]['mhalo'] for i, item in enumerate(GalA1_history)]

    # GAR_l = [GalA1_history[i]['GAR'] for i, item in enumerate(GalA1_history)]
    # sMIR_l = [GalA1_history[i]['sMIR'] for i, item in enumerate(GalA1_history)]
    # rsSFR_l = [GalA1_history[i]['rsSFR'] for i, item in enumerate(GalA1_history)]

    GCR_l = [GalA1_history[i]['GCR'] for i, item in enumerate(GalA1_history)]
    SFR_l = [GalA1_history[i]['SFR'] for i, item in enumerate(GalA1_history)]
    MLR_l = [GalA1_history[i]['MLR'] for i, item in enumerate(GalA1_history)]
    MIR_l = [GalA1_history[i]['MIR'] for i, item in enumerate(GalA1_history)]

    # fout_l = [GalA1_history[i]['fout'] for i, item in enumerate(GalA1_history)]
    # fgas_l = [GalA1_history[i]['fgas'] for i, item in enumerate(GalA1_history)]
    # fstar_l = [GalA1_history[i]['fstar'] for i, item in enumerate(GalA1_history)]

    # ========
    # plotting

    fig_masses, ax_masses = plt.subplots(figsize=(9,6))
    ax_masses.plot(z_l[5:], mhalo_l[5:], label="mhalo (halo mass)", c="xkcd:red")
    ax_masses.plot(z_l[5:], mgas_l[5:], label="mgas (gas mass)", c="xkcd:blue")
    ax_masses.plot(z_l[5:], mstar_l[5:], label="mstar (stellar mass)", c="xkcd:orange")
    ax_masses.plot(z_l[5:], mout_l[5:], label="mout (ejected masss)", c="xkcd:green")
    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Redshift')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')
    ax_masses.legend()
    fig_masses.show()


    fig_rates, ax_rates = plt.subplots(figsize=(9,6))
    ax_rates.plot(z_l[5:], MIR_l[5:], label="MIR (mass increase)", c="xkcd:red")
    ax_rates.plot(z_l[5:], GCR_l[5:], label="GCR (gas change)", c="xkcd:blue")
    ax_rates.plot(z_l[5:], SFR_l[5:], label="SFR (star formation)", c="xkcd:orange")
    ax_rates.plot(z_l[5:], MLR_l[5:], label="MLR (mass loss)", c="xkcd:green")

    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Redshift')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')
    ax_rates.legend()
    fig_rates.show()

    return


def testrun_5(with_burnin = False, timestep: float = 1.e-3, silent: bool = False):
    EnvA = sgm.Environment(name='Environment A', zstart=2.)
    if not silent:
        print(f"Created '{EnvA.name}'")
    HaloA1 = EnvA.create_Halo(name='Halo A1', mgas=0., mdm=10**12.5)
    GalA1 = HaloA1.create_Galaxy(with_burnin=with_burnin,
                                 with_burnin_dict={'cycle_steps': 5,
                                                   'fixed_attr': ['mhalo', 'mstar'],
                                                   'check_attr': ['mgas', 'SFR', 'GCR', 'MLR']
                                                   },
                                 name='Galaxy A1',
                                 mgas=10**9.5,
                                 mstar=10**10.5
                                 )
    if not silent:
        print(f"Created '{GalA1.name}'")
        print(EnvA)
        print(GalA1)

    if not silent:
        print("Running a simulation...")
    GalA1_history = []
    GalA1_history.append(GalA1.make_snapshot())
    HaloA1_history = []
    HaloA1_history.append(HaloA1.make_snapshot())
    if not silent:
        print(f"{0}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")
    for t in range(10500):  # from lookbacktime at z=2
        EnvA.evolve(timestep=timestep, mode='intuitive')  # combine this line and the next?
        GalA1_history.append(GalA1.make_snapshot())
        HaloA1_history.append(HaloA1.make_snapshot())
        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    if not silent:
        print(EnvA)
        print(HaloA1)
        print(GalA1)


    # timesteps list for plotting
    timesteps_l = np.linspace(0, len(GalA1_history) * timestep, len(GalA1_history))
    lookbacktime_l = [GalA1_history[i]['lookbacktime'] for i, item in enumerate(GalA1_history)]

    # grab the data we want to plot from history of galaxy
    z_l = [GalA1_history[i]['z'] for i, item in enumerate(GalA1_history)]

    mgas_l = [GalA1_history[i]['mgas'] for i, item in enumerate(GalA1_history)]
    mstar_l = [GalA1_history[i]['mstar'] for i, item in enumerate(GalA1_history)]
    mout_l = [GalA1_history[i]['mout'] for i, item in enumerate(GalA1_history)]
    mhalo_l = [HaloA1_history[i]['mtot'] for i, item in enumerate(HaloA1_history)]

    # GAR_l = [GalA1_history[i]['GAR'] for i, item in enumerate(GalA1_history)]
    # sMIR_l = [GalA1_history[i]['sMIR'] for i, item in enumerate(GalA1_history)]
    # rsSFR_l = [GalA1_history[i]['rsSFR'] for i, item in enumerate(GalA1_history)]

    GCR_l = [GalA1_history[i]['GCR'] for i, item in enumerate(GalA1_history)]
    SFR_l = [GalA1_history[i]['SFR'] for i, item in enumerate(GalA1_history)]
    MLR_l = [GalA1_history[i]['MLR'] for i, item in enumerate(GalA1_history)]
    MIR_l = [HaloA1_history[i]['MIR'] for i, item in enumerate(HaloA1_history)]

    # fout_l = [GalA1_history[i]['fout'] for i, item in enumerate(GalA1_history)]
    # fgas_l = [GalA1_history[i]['fgas'] for i, item in enumerate(GalA1_history)]
    # fstar_l = [GalA1_history[i]['fstar'] for i, item in enumerate(GalA1_history)]

    # ========
    # plotting

    fig_masses, ax_masses = plt.subplots(figsize=(9,6))
    ax_masses.plot(lookbacktime_l[5:], mhalo_l[5:], label="mhalo (total halo mass)", c="xkcd:red")
    ax_masses.plot(lookbacktime_l[5:], mgas_l[5:], label="mgas (gas mass)", c="xkcd:blue")
    ax_masses.plot(lookbacktime_l[5:], mstar_l[5:], label="mstar (stellar mass)", c="xkcd:orange")
    ax_masses.plot(lookbacktime_l[5:], mout_l[5:], label="mout (ejected masss)", c="xkcd:green")
    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Lookbacktime (Gyr)')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')
    ax_masses.legend()
    fig_masses.show()


    fig_rates, ax_rates = plt.subplots(figsize=(9,6))
    ax_rates.plot(lookbacktime_l[5:], MIR_l[5:], label="MIR (mass increase)", c="xkcd:red")
    ax_rates.plot(lookbacktime_l[5:], GCR_l[5:], label="GCR (gas change)", c="xkcd:blue")
    ax_rates.plot(lookbacktime_l[5:], SFR_l[5:], label="SFR (star formation)", c="xkcd:orange")
    ax_rates.plot(lookbacktime_l[5:], MLR_l[5:], label="MLR (mass loss)", c="xkcd:green")

    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Lookbacktime (Gyr)')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')
    ax_rates.legend()
    fig_rates.show()

    return


def test_fgal1(with_burnin = False, timestep: float = 1.e-3, silent: bool = False):
    EnvA = sgm.Environment(name='Environment A', zstart=6.)

    HaloA_IC_l = [{'name': f'Halo A{i}',
                   'mgas': 10**10,
                   'mdm': 10**12.5} for i in range(5)]

    GalA_IC_l = [{'name': f'Gal A{i}',
                  'fgal': 0.1 * (i+1),
                  'mgas': 10**9.5,
                  'mstar': 10**10.5} for i in range(5)]

    HaloA_l = [EnvA.create_Halo(**IC) for IC in HaloA_IC_l]

    GalA_l = [halo.create_Galaxy(**IC) for halo, IC in zip(HaloA_l, GalA_IC_l)]


    if not silent:
        print(EnvA)
        for halo, gal in zip(HaloA_l, GalA_l):
            print(halo)
            print(gal)

    if not silent:
        print("Running a simulation...")

    HaloA_history_l = []*len(HaloA_l)
    for halo in HaloA_l:   # make first snapshot (capturing ICs)
        HaloA_history_l.append([halo.make_snapshot().data])

    GalA_history_l = []*len(GalA_l)
    for gal in GalA_l:   # make first snapshot (capturing ICs)
        GalA_history_l.append([gal.make_snapshot().data])


    if not silent:
        print(f"{0}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")
    for t in range(12857):  # from lookbacktime at z=6
    # for t in range(101):  # from lookbacktime at z=6
        EnvA.evolve(timestep=timestep, mode='intuitive')

        for halo, halo_hist in zip(HaloA_l, HaloA_history_l):   # make snapshot of each step
            halo_hist.append(halo.make_snapshot().data)

        for gal, gal_hist in zip(GalA_l, GalA_history_l):   # make first snapshot (capturing ICs)
            gal_hist.append(gal.make_snapshot().data)

        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    if not silent:
        print(EnvA)
        for halo, gal in zip(HaloA_l, GalA_l):
            print(halo)
            print(gal)


    # timesteps list for plotting
    timesteps_l = np.linspace(0, len(GalA_history_l[0]) * timestep, len(GalA_history_l[0]))

    # grab the data we want to plot from history of galaxy
    lookbacktime_l = [gal['lookbacktime'] for gal in GalA_history_l[0]]
    z_l = [gal['z'] for gal in GalA_history_l[0]]

    HaloA_plotting_l = [] * len(HaloA_history_l)
    GalA_plotting_l = [] * len(GalA_history_l)
    for (halo_hist, gal_hist) in zip(HaloA_history_l, GalA_history_l):
        GalA_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                                 'mstar': [gal['mstar'] for gal in gal_hist],
                                 'mout': [gal['mout'] for gal in gal_hist],
                                 'GCR': [gal['GCR'] for gal in gal_hist],
                                 'SFR': [gal['SFR'] for gal in gal_hist],
                                 'MLR': [gal['MLR'] for gal in gal_hist]})
        HaloA_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                  'MIR': [halo['MIR'] for halo in halo_hist]})

    # GAR_l = [GalA1_history[i]['GAR'] for i, item in enumerate(GalA1_history)]
    # sMIR_l = [GalA1_history[i]['sMIR'] for i, item in enumerate(GalA1_history)]
    # rsSFR_l = [GalA1_history[i]['rsSFR'] for i, item in enumerate(GalA1_history)]

    # fout_l = [GalA1_history[i]['fout'] for i, item in enumerate(GalA1_history)]
    # fgas_l = [GalA1_history[i]['fgas'] for i, item in enumerate(GalA1_history)]
    # fstar_l = [GalA1_history[i]['fstar'] for i, item in enumerate(GalA1_history)]

    # ========
    # plotting

    # # expand colour cycle
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    linestyles = ['--'] * len(GalA_plotting_l)
    linestyles[0] = '-.'
    linestyles[-1] = '-'

    fig_masses, ax_masses = plt.subplots(figsize=(10,6))

    for gal_plot, halo_plot, gal_IC, ls in zip(GalA_plotting_l, HaloA_plotting_l, GalA_IC_l, linestyles):
        fgal=gal_IC['fgal']
        ax_masses.plot(lookbacktime_l[2:], halo_plot['mtot'][2:], label=f"mtot (fgal={fgal:.2})", c="xkcd:red", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mgas'][2:], label=f"mgas (fgal={fgal:.2})", c="xkcd:blue", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mstar'][2:], label=f"mstar (fgal={fgal:.2})", c="xkcd:orange", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mout'][2:], label=f"mout (fgal={fgal:.2})", c="xkcd:green", ls=ls)

    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Lookbacktime (Gyr)')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')
    box = ax_masses.get_position()
    ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig_masses.show()


    fig_rates, ax_rates = plt.subplots(figsize=(10,6))

    for gal_plot, halo_plot, gal_IC, ls in zip(GalA_plotting_l, HaloA_plotting_l, GalA_IC_l, linestyles):
        fgal=gal_IC['fgal']
        ax_rates.plot(lookbacktime_l[2:], halo_plot['MIR'][2:], label=f"MIR (fgal={fgal:.2})", c="xkcd:red", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['GCR'][2:], label=f"GCR (fgal={fgal:.2})", c="xkcd:blue", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['SFR'][2:], label=f"SFR (fgal={fgal:.2})", c="xkcd:orange", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['MLR'][2:], label=f"MLR (fgal={fgal:.2})", c="xkcd:green", ls=ls)


    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Lookbacktime (Gyr)')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')
    box = ax_rates.get_position()
    ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig_rates.show()

    return


def test_sfe1(with_burnin = False,
              zstart: float = 2.,
              timestep: float = 1.e-3,
              varythis: str = 'SFE',
              varythis_val: tuple = (0.01, 0.05, 0.10, 0.15, 0.20),
              silent: bool = False):

    lookbacktime = cosmo.lookback_time(zstart).value
    n_steps = int(np.floor(lookbacktime / timestep))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    EnvA = sgm.Environment(name='Environment_A', zstart=zstart)

    HaloA_IC_l = [{'name': f'Halo_A{i}',
                   'mgas': 10**10,
                   'mdm': 10**12.5} for i, val in enumerate(varythis_val)]

    GalA_IC_l = [{'name': f'Galaxy_A{i}',
                  'fgal': 0.05,
                  'mgas': 10**9.5,
                  'mstar': 10**10.5,
                  varythis: val} for i, val in enumerate(varythis_val)]

    HaloA_l = [EnvA.create_Halo(**IC) for IC in HaloA_IC_l]

    GalA_l = [halo.create_Galaxy(**IC) for halo, IC in zip(HaloA_l, GalA_IC_l)]


    if not silent:
        print(EnvA)
        for halo, gal in zip(HaloA_l, GalA_l):
            print(halo)
            print(gal)

    if not silent:
        print("Running a simulation...")

    HaloA_history_l = []*len(HaloA_l)
    for halo in HaloA_l:   # make first snapshot (capturing ICs)
        HaloA_history_l.append([halo.make_snapshot().data])

    GalA_history_l = []*len(GalA_l)
    for gal in GalA_l:   # make first snapshot (capturing ICs)
        GalA_history_l.append([gal.make_snapshot().data])


    if not silent:
        print(f"{0}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    for t in range(n_steps):  # from lookbacktime at z=2, only 1/10 of prev. resolution
    # for t in range(12857):  # from lookbacktime at z=2
    # for t in range(101):  # from lookbacktime at z=6
        EnvA.evolve(timestep=timestep, mode='intuitive')

        for halo, halo_hist in zip(HaloA_l, HaloA_history_l):   # make snapshot of each step
            halo_hist.append(halo.make_snapshot().data)

        for gal, gal_hist in zip(GalA_l, GalA_history_l):   # make first snapshot (capturing ICs)
            gal_hist.append(gal.make_snapshot().data)

        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    # if not silent:
    EnvA_final_fp = EnvA.make_snapshot().save_to_disk(f'outputs/{EnvA.name}_final.json')
    print(EnvA_final_fp)

    HaloA_final_fp_l = []
    GalA_final_fp_l = []
    for halo, gal in zip(HaloA_l, GalA_l):
        HaloA_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/{halo.name}_final.json'))
        print(HaloA_final_fp_l[-1])

        GalA_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/{gal.name}_final.json'))
        print(GalA_final_fp_l[-1])



    # ======================
    # grab data for plotting


    # timesteps list for plotting
    timesteps_l = np.linspace(0, lookbacktime, n_steps)

    # grab the data we want to plot from history of galaxy
    lookbacktime_l = [gal['lookbacktime'] for gal in GalA_history_l[0]]
    z_l = [gal['z'] for gal in GalA_history_l[0]]

    HaloA_plotting_l = [] * len(HaloA_history_l)
    GalA_plotting_l = [] * len(GalA_history_l)
    for (halo_hist, gal_hist) in zip(HaloA_history_l, GalA_history_l):
        GalA_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                                'mstar': [gal['mstar'] for gal in gal_hist],
                                'mout': [gal['mout'] for gal in gal_hist],
                                'GCR': [gal['GCR'] for gal in gal_hist],
                                'SFR': [gal['SFR'] for gal in gal_hist],
                                'MLR': [gal['MLR'] for gal in gal_hist]})
        HaloA_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                 'MIR': [halo['MIR'] for halo in halo_hist]})

    # GAR_l = [GalA1_history[i]['GAR'] for i, item in enumerate(GalA1_history)]
    # sMIR_l = [GalA1_history[i]['sMIR'] for i, item in enumerate(GalA1_history)]
    # rsSFR_l = [GalA1_history[i]['rsSFR'] for i, item in enumerate(GalA1_history)]

    # fout_l = [GalA1_history[i]['fout'] for i, item in enumerate(GalA1_history)]
    # fgas_l = [GalA1_history[i]['fgas'] for i, item in enumerate(GalA1_history)]
    # fstar_l = [GalA1_history[i]['fstar'] for i, item in enumerate(GalA1_history)]

    # ========
    # plotting

    # # expand colour cycle
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    linestyles = ['--'] * len(GalA_plotting_l)
    linestyles[0] = '-.'
    linestyles[-1] = '-'

    fig_masses, ax_masses = plt.subplots(figsize=(10,6))

    for gal_plot, halo_plot, gal_IC, ls in zip(GalA_plotting_l, HaloA_plotting_l, GalA_IC_l, linestyles):
        varythis_val=gal_IC[varythis]
        ax_masses.plot(lookbacktime_l[2:], halo_plot['mtot'][2:], label=f"mtot ({varythis}={varythis_val:.2})", c="xkcd:red", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mgas'][2:], label=f"mgas ({varythis}={varythis_val:.2})", c="xkcd:blue", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mstar'][2:], label=f"mstar ({varythis}={varythis_val:.2})", c="xkcd:orange", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mout'][2:], label=f"mout ({varythis}={varythis_val:.2})", c="xkcd:green", ls=ls)

    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Lookbacktime (Gyr)')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')
    box = ax_masses.get_position()
    ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig_masses.show()


    fig_rates, ax_rates = plt.subplots(figsize=(10,6))

    for gal_plot, halo_plot, gal_IC, ls in zip(GalA_plotting_l, HaloA_plotting_l, GalA_IC_l, linestyles):
        varythis_val=gal_IC[varythis]
        ax_rates.plot(lookbacktime_l[2:], halo_plot['MIR'][2:], label=f"MIR ({varythis}={varythis_val:.2})", c="xkcd:red", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['GCR'][2:], label=f"GCR ({varythis}={varythis_val:.2})", c="xkcd:blue", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['SFR'][2:], label=f"SFR ({varythis}={varythis_val:.2})", c="xkcd:orange", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['MLR'][2:], label=f"MLR ({varythis}={varythis_val:.2})", c="xkcd:green", ls=ls)


    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Lookbacktime (Gyr)')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')
    box = ax_rates.get_position()
    ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig_rates.show()

    return


def test_multiple(with_burnin = False,
                  zstart: float = 2.,
                  timestep: float = 1.e-3,
                  vary_params = None,
                  silent: bool = False):

    vary_params = {
        'fgal': (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        'SFE': (0.01, 0.05, 0.10, 0.15, 0.20),
        'HLF': (0.01, 0.05, 0.10, 0.20)
    } if vary_params is None else vary_params

    # calc. lookback time and number of steps from zstart and timestep
    lookbacktime = cosmo.lookback_time(zstart).value
    n_steps = int(np.floor(lookbacktime / timestep))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    # ------------------------------------------
    # make grid with possible value combinations
    vary_n = len(vary_params)
    vary_list = [v for v in vary_params.values()]

    # generate empty array of dimensions to fit the n params in the innermost part (in all j1*j2*...jn combinations)
    vary_grid_dims = tuple(len(vals) for vals in vary_list + [vary_list])
    vary_grid = np.full(shape=vary_grid_dims, fill_value=np.nan)

    for i, vals in enumerate(vary_list):
        # pull one axis after the other to the forefront (rollaxis returns just a view)
        _grid_view = np.rollaxis(vary_grid, i)

        # iterate over the different possible values for this vals (a slice from vary_param)
        for grid_line, v in zip(_grid_view, vals):
            grid_line[..., i] = v

    # ---------------------------------------
    # create environment, halos and galaxies based on the vary_grid array

    # set up arrays for Halos and Galaxies
    obj_grid_dims = tuple(len(vals) for vals in vary_list)
    halo_grid = np.empty(shape=obj_grid_dims, dtype=object)
    galaxy_grid = np.empty_like(halo_grid, dtype=object)

    # create flat views on the object arrays and for the parameter-tuples array
    vary_grid_flat = vary_grid.reshape(-1, vary_grid.shape[-1])
    # halo_grid_flat = halo_grid.flat
    # galaxy_grid_flat = galaxy_grid.flat

    # create environment
    env = sgm.Environment(name='Environment', zstart=zstart)

    # create halos and corresponding galaxies
    for i, valset in enumerate(vary_grid_flat):
        # ASSIGNMENT of vary-values IS STILL HARD-CODED!!!!!!!
        halo_grid.flat[i] = env.create_Halo(
            name= f'Halo_{i}',
            mgas= 10**10,
            mdm= 10**12.5,
            HLF= valset[3]
        )
        galaxy_grid.flat[i] = halo_grid.flat[i].create_Galaxy(
            name= f'Galaxy_{i}',
            mgas= 10**9.5,
            mstar= 10**10.5,
            fgal=valset[0],
            SFE=valset[1],
            MLF=valset[2]
        )


    if not silent:
        print(env)
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            print(halo)
            print(gal)

    if not silent:
        print("Running a simulation...")

    halo_history_l = []*len(halo_grid.flat)
    for halo in halo_grid.flat:   # make first snapshot (capturing ICs)
        halo_history_l.append([halo.make_snapshot().data])

    gal_history_l = []*len(galaxy_grid.flat)
    for gal in galaxy_grid.flat:   # make first snapshot (capturing ICs)
        gal_history_l.append([gal.make_snapshot().data])


    if not silent:
        print(f"{0}, z={env.z}, lookbacktime={env.lookbacktime}")

    for t in trange(n_steps):  # from lookbacktime at z=2, only 1/10 of prev. resolution
        # for t in range(12857):  # from lookbacktime at z=2
        # for t in range(101):  # from lookbacktime at z=6
        env.evolve(timestep=timestep, mode='intuitive')

        for halo, halo_hist in zip(halo_grid.flat, halo_history_l):   # make snapshot of each step
            halo_hist.append(halo.make_snapshot().data)

        for gal, gal_hist in zip(galaxy_grid.flat, gal_history_l):   # make first snapshot (capturing ICs)
            gal_hist.append(gal.make_snapshot().data)

        if not silent:
            print(f"{t+1}, z={env.z}, lookbacktime={env.lookbacktime}")

    # if not silent:
    env_final_fp = env.make_snapshot().save_to_disk(f'outputs/{env.name}_final.json')
    print(env_final_fp)

    halo_final_fp_l = []
    gal_final_fp_l = []
    for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
        halo_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/{halo.name}_final.json'))
        print(halo_final_fp_l[-1])

        gal_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/{gal.name}_final.json'))
        print(gal_final_fp_l[-1])



    # ======================
    # grab data for plotting


    # timesteps list for plotting
    timesteps_l = np.linspace(0, lookbacktime, n_steps)

    # grab the data we want to plot from history of galaxy
    lookbacktime_l = [gal['lookbacktime'] for gal in gal_history_l[0]]
    z_l = [gal['z'] for gal in gal_history_l[0]]

    halo_plotting_l = [] * len(halo_history_l)
    gal_plotting_l = [] * len(gal_history_l)
    for (halo_hist, gal_hist) in zip(halo_history_l, gal_history_l):
        gal_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                                'mstar': [gal['mstar'] for gal in gal_hist],
                                'mout': [gal['mout'] for gal in gal_hist],
                                'GCR': [gal['GCR'] for gal in gal_hist],
                                'SFR': [gal['SFR'] for gal in gal_hist],
                                'MLR': [gal['MLR'] for gal in gal_hist]})
        halo_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                 'MIR': [halo['MIR'] for halo in halo_hist]})

    # ========
    # plotting

    # # expand colour cycle
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    linestyles = ['--'] * len(gal_plotting_l)
    linestyles[0] = '-.'
    linestyles[-1] = '-'

    # fig_masses, ax_masses = plt.subplots(figsize=(10, 6))
    fig_masses, ax_masses = plt.subplots(figsize=(9, 6))

    for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
        # varythis_val=vary_vals[varythis]

        vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

        ax_masses.plot(lookbacktime_l[2:], halo_plot['mtot'][2:], label=f"mtot ({vary_str})", c="xkcd:red", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mgas'][2:], label=f"mgas ({vary_str})", c="xkcd:blue", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mstar'][2:], label=f"mstar ({vary_str})", c="xkcd:orange", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mout'][2:], label=f"mout ({vary_str})", c="xkcd:green", ls=ls)

    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Lookbacktime (Gyr)')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')

    # # out legend outside the axes
    # box = ax_masses.get_position()
    # ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig_masses.show()


    # fig_rates, ax_rates = plt.subplots(figsize=(10,6))
    fig_rates, ax_rates = plt.subplots(figsize=(9,6))

    for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
        # varythis_val=vary_vals[varythis]

        vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

        ax_rates.plot(lookbacktime_l[2:], halo_plot['MIR'][2:], label=f"MIR ({vary_str})", c="xkcd:red", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['GCR'][2:], label=f"GCR ({vary_str})", c="xkcd:blue", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['SFR'][2:], label=f"SFR ({vary_str})", c="xkcd:orange", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['MLR'][2:], label=f"MLR ({vary_str})", c="xkcd:green", ls=ls)


    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Lookbacktime (Gyr)')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')

    # # put legend outside the axes
    # box = ax_rates.get_position()
    # ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig_rates.show()

    return


def test_multiple_evolution(with_burnin = False,
                            zstart: float = 2.,
                            timestep: float = 1.e-3,
                            vary_params = None,
                            write_todisk: int = 10,
                            silent: bool = False):

    vary_params = {
        'fgal': (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        'SFE': (0.01, 0.05, 0.10, 0.15, 0.20),
        'HLF': (0.01, 0.05, 0.10, 0.20)
    } if vary_params is None else vary_params

    # calc. lookback time and number of steps from zstart and timestep
    lookbacktime = cosmo.lookback_time(zstart).value
    n_steps = int(np.floor(lookbacktime / timestep))
    # n_steps_maxlen = len(str(n_steps))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    # ------------------------------------------
    # make grid with possible value combinations
    vary_n = len(vary_params)
    vary_list = [v for v in vary_params.values()]

    # generate empty array of dimensions to fit the n params in the innermost part (in all j1*j2*...jn combinations)
    vary_grid_dims = tuple(len(vals) for vals in vary_list + [vary_list])
    vary_grid = np.full(shape=vary_grid_dims, fill_value=np.nan)

    for i, vals in enumerate(vary_list):
        # pull one axis after the other to the forefront (rollaxis returns just a view)
        _grid_view = np.rollaxis(vary_grid, i)

        # iterate over the different possible values for this vals (a slice from vary_param)
        for grid_line, v in zip(_grid_view, vals):
            grid_line[..., i] = v

    # ---------------------------------------
    # create environment, halos and galaxies based on the vary_grid array

    # set up arrays for Halos and Galaxies
    obj_grid_dims = tuple(len(vals) for vals in vary_list)
    halo_grid = np.empty(shape=obj_grid_dims, dtype=object)
    galaxy_grid = np.empty_like(halo_grid, dtype=object)

    # create flat views on the object arrays and for the parameter-tuples array
    vary_grid_flat = vary_grid.reshape(-1, vary_grid.shape[-1])
    # halo_grid_flat = halo_grid.flat
    # galaxy_grid_flat = galaxy_grid.flat

    # create environment
    env = sgm.Environment(name='Environment', zstart=zstart)

    # create halos and corresponding galaxies
    for i, valset in enumerate(vary_grid_flat):
        # ASSIGNMENT of vary-values IS STILL HARD-CODED!!!!!!!
        halo_grid.flat[i] = env.create_Halo(
            name= f'Halo_{i}',
            mgas= 10**10,
            mdm= 10**12.5,
            HLF= valset[3]
        )
        galaxy_grid.flat[i] = halo_grid.flat[i].create_Galaxy(
            name= f'Galaxy_{i}',
            # mgas= 10**9.5,
            # mstar= 10**10.5,
            mgas= valset[4]/10,
            mstar= valset[4],
            fgal=valset[0],
            SFE=valset[1],
            MLF=valset[2]
        )


    if not silent:
        print(env)
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            print(halo)
            print(gal)

    if not silent:
        print("Running a simulation...")

    halo_history_l = []*len(halo_grid.flat)
    for halo in halo_grid.flat:   # make first snapshot (capturing ICs)
        snp = halo.make_snapshot()
        halo_history_l.append([snp.data])
        # writing to disk
        if write_todisk > 0:
            snp.save_to_disk(f'outputs/{snp.autoname_with_index(0, n_steps)}')

    gal_history_l = []*len(galaxy_grid.flat)
    for gal in galaxy_grid.flat:   # make first snapshot (capturing ICs)
        snp = gal.make_snapshot()
        gal_history_l.append([snp.data])
        if write_todisk > 0:
            snp.save_to_disk(f'outputs/{snp.autoname_with_index(0, n_steps)}')


    if not silent:
        print(f"{0}, z={env.z}, lookbacktime={env.lookbacktime}")

    for t in trange(1, n_steps+1):  # from lookbacktime at z=2, only 1/10 of prev. resolution
        # for t in range(12857):  # from lookbacktime at z=2
        # for t in range(101):  # from lookbacktime at z=6
        env.evolve(timestep=timestep, mode='intuitive')


        for halo, halo_hist in zip(halo_grid.flat, halo_history_l):   # make snapshot of each step
            snp = halo.make_snapshot()
            halo_hist.append(snp.data)
            if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                snp.save_to_disk(f'outputs/{snp.autoname_with_index(t, n_steps)}')

        for gal, gal_hist in zip(galaxy_grid.flat, gal_history_l):   # make first snapshot (capturing ICs)
            snp = gal.make_snapshot()
            gal_hist.append(snp.data)
            if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                snp.save_to_disk(f'outputs/{snp.autoname_with_index(t, n_steps)}')

        if not silent:
            print(f"{t+1}, z={env.z}, lookbacktime={env.lookbacktime}")

    # if not silent:
    env_final_fp = env.make_snapshot().save_to_disk(f'outputs/{env.name}_final.json')
    print(env_final_fp)

    halo_final_fp_l = []
    gal_final_fp_l = []
    for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
        halo_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/{halo.name}_final.json'))
        print(halo_final_fp_l[-1])

        gal_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/{gal.name}_final.json'))
        print(gal_final_fp_l[-1])



    # ======================
    # grab data for plotting


    # timesteps list for plotting
    timesteps_l = np.linspace(0, lookbacktime, n_steps)

    # grab the data we want to plot from history of galaxy
    lookbacktime_l = [gal['lookbacktime'] for gal in gal_history_l[0]]
    z_l = [gal['z'] for gal in gal_history_l[0]]

    halo_plotting_l = [] * len(halo_history_l)
    gal_plotting_l = [] * len(gal_history_l)
    for (halo_hist, gal_hist) in zip(halo_history_l, gal_history_l):
        gal_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                               'mstar': [gal['mstar'] for gal in gal_hist],
                               'mout': [gal['mout'] for gal in gal_hist],
                               'GCR': [gal['GCR'] for gal in gal_hist],
                               'SFR': [gal['SFR'] for gal in gal_hist],
                               'MLR': [gal['MLR'] for gal in gal_hist]})
        halo_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                'MIR': [halo['MIR'] for halo in halo_hist]})

    # ========
    # plotting

    # # expand colour cycle
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    linestyles = ['--'] * len(gal_plotting_l)
    linestyles[0] = '-.'
    linestyles[-1] = '-'

    # fig_masses, ax_masses = plt.subplots(figsize=(10, 6))
    fig_masses, ax_masses = plt.subplots(figsize=(9, 6))

    for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
        # varythis_val=vary_vals[varythis]

        vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

        ax_masses.plot(lookbacktime_l[2:], halo_plot['mtot'][2:], label=f"mtot ({vary_str})", c="xkcd:red", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mgas'][2:], label=f"mgas ({vary_str})", c="xkcd:blue", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mstar'][2:], label=f"mstar ({vary_str})", c="xkcd:orange", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mout'][2:], label=f"mout ({vary_str})", c="xkcd:green", ls=ls)

    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Lookbacktime (Gyr)')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')

    # # out legend outside the axes
    # box = ax_masses.get_position()
    # ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig_masses.show()


    # fig_rates, ax_rates = plt.subplots(figsize=(10,6))
    fig_rates, ax_rates = plt.subplots(figsize=(9,6))

    for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
        # varythis_val=vary_vals[varythis]

        vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

        ax_rates.plot(lookbacktime_l[2:], halo_plot['MIR'][2:], label=f"MIR ({vary_str})", c="xkcd:red", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['GCR'][2:], label=f"GCR ({vary_str})", c="xkcd:blue", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['SFR'][2:], label=f"SFR ({vary_str})", c="xkcd:orange", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['MLR'][2:], label=f"MLR ({vary_str})", c="xkcd:green", ls=ls)


    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Lookbacktime (Gyr)')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')

    # # put legend outside the axes
    # box = ax_rates.get_position()
    # ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig_rates.show()

    return


def test_multiple_evolution_timedirection(
        with_burnin = False,
        zstart: float = 2.,
        zend: float = 0.,
        timestep: float = 1.e-3,
        vary_params = None,
        write_todisk: int = 10,
        silent: bool = False):

    vary_params = {
        'fgal': (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        'SFE': (0.01, 0.05, 0.10, 0.15, 0.20),
        'HLF': (0.01, 0.05, 0.10, 0.20)
    } if vary_params is None else vary_params

    # calc. lookback time and number of steps from zstart and timestep
    lookbacktime = cosmo.lookback_time(zstart).value
    endtime = cosmo.lookback_time(zend).value
    if timestep != 0:
        n_steps = int(np.floor((lookbacktime - endtime) / timestep))    # works for both directions in time!
    else:
        n_steps = 1     # WARNING: this is a trivial case with just one timestep running
        print("WARNING: zstart = zend, so only one timestep will be cpmputed!")
    # n_steps_maxlen = len(str(n_steps))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'running until z={zend:.3}')
    print(f'corresponding lookback time is {endtime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    # ------------------------------------------
    # make grid with possible value combinations
    vary_n = len(vary_params)
    vary_list = [v for v in vary_params.values()]

    # generate empty array of dimensions to fit the n params in the innermost part (in all j1*j2*...jn combinations)
    vary_grid_dims = tuple(len(vals) for vals in vary_list + [vary_list])
    vary_grid = np.full(shape=vary_grid_dims, fill_value=np.nan)

    for i, vals in enumerate(vary_list):
        # pull one axis after the other to the forefront (rollaxis returns just a view)
        _grid_view = np.rollaxis(vary_grid, i)

        # iterate over the different possible values for this vals (a slice from vary_param)
        for grid_line, v in zip(_grid_view, vals):
            grid_line[..., i] = v

    # ---------------------------------------
    # create environment, halos and galaxies based on the vary_grid array

    # set up arrays for Halos and Galaxies
    obj_grid_dims = tuple(len(vals) for vals in vary_list)
    halo_grid = np.empty(shape=obj_grid_dims, dtype=object)
    galaxy_grid = np.empty_like(halo_grid, dtype=object)

    # create flat views on the object arrays and for the parameter-tuples array
    vary_grid_flat = vary_grid.reshape(-1, vary_grid.shape[-1])
    # halo_grid_flat = halo_grid.flat
    # galaxy_grid_flat = galaxy_grid.flat

    # create environment
    env = sgm.Environment(name='Environment', zstart=zstart)

    # create halos and corresponding galaxies
    for i, valset in enumerate(vary_grid_flat):
        # ASSIGNMENT of vary-values IS STILL HARD-CODED!!!!!!!
        halo_grid.flat[i] = env.create_Halo(
            name= f'Halo_{i}',
            mgas= 10**10,
            mdm= 10**12.5,
            HLF= valset[3]
        )
        galaxy_grid.flat[i] = halo_grid.flat[i].create_Galaxy(
            name= f'Galaxy_{i}',
            # mgas= 10**9.5,
            # mstar= 10**10.5,
            mgas= valset[4]/10,
            mstar= valset[4],
            fgal=valset[0],
            SFE=valset[1],
            MLF=valset[2]
        )


    if not silent:
        print(env)
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            print(halo)
            print(gal)

    if not silent:
        print("Running a simulation...")

    halo_history_l = []*len(halo_grid.flat)
    for halo in halo_grid.flat:   # make first snapshot (capturing ICs)
        snp = halo.make_snapshot()
        halo_history_l.append([snp.data])
        # writing to disk
        if write_todisk > 0:
            snp.save_to_disk(f'outputs/_tmp/{snp.autoname_with_index(0, n_steps)}')

    gal_history_l = []*len(galaxy_grid.flat)
    for gal in galaxy_grid.flat:   # make first snapshot (capturing ICs)
        snp = gal.make_snapshot()
        gal_history_l.append([snp.data])
        if write_todisk > 0:
            snp.save_to_disk(f'outputs/_tmp/{snp.autoname_with_index(0, n_steps)}')


    if not silent:
        print(f"{0}, z={env.z}, lookbacktime={env.lookbacktime}")

    for t in trange(1, n_steps+1):  # from lookbacktime at z=2, only 1/10 of prev. resolution
        # for t in range(12857):  # from lookbacktime at z=2
        # for t in range(101):  # from lookbacktime at z=6
        env.evolve(timestep=timestep, mode='intuitive')


        for halo, halo_hist in zip(halo_grid.flat, halo_history_l):   # make snapshot of each step
            snp = halo.make_snapshot()
            halo_hist.append(snp.data)
            if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                snp.save_to_disk(f'outputs/_tmp/{snp.autoname_with_index(t, n_steps)}')

        for gal, gal_hist in zip(galaxy_grid.flat, gal_history_l):   # make first snapshot (capturing ICs)
            snp = gal.make_snapshot()
            gal_hist.append(snp.data)
            if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                snp.save_to_disk(f'outputs/_tmp/{snp.autoname_with_index(t, n_steps)}')

        if not silent:
            print(f"{t+1}, z={env.z}, lookbacktime={env.lookbacktime}")

    # if not silent:
    env_final_fp = env.make_snapshot().save_to_disk(f'outputs/_tmp/{env.name}_final.json')
    print(env_final_fp)

    halo_final_fp_l = []
    gal_final_fp_l = []
    for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
        halo_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/_tmp/{halo.name}_final.json'))
        print(halo_final_fp_l[-1])

        gal_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/_tmp/{gal.name}_final.json'))
        print(gal_final_fp_l[-1])



    # ======================
    # grab data for plotting


    # timesteps list for plotting
    timesteps_l = np.linspace(0, lookbacktime, n_steps)

    # grab the data we want to plot from history of galaxy
    lookbacktime_l = [gal['lookbacktime'] for gal in gal_history_l[0]]
    z_l = [gal['z'] for gal in gal_history_l[0]]

    halo_plotting_l = [] * len(halo_history_l)
    gal_plotting_l = [] * len(gal_history_l)
    for (halo_hist, gal_hist) in zip(halo_history_l, gal_history_l):
        gal_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                               'mstar': [gal['mstar'] for gal in gal_hist],
                               'mout': [gal['mout'] for gal in gal_hist],
                               'GCR': [gal['GCR'] for gal in gal_hist],
                               'SFR': [gal['SFR'] for gal in gal_hist],
                               'MLR': [gal['MLR'] for gal in gal_hist]})
        halo_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                'MIR': [halo['MIR'] for halo in halo_hist]})

    # ========
    # plotting

    # # expand colour cycle
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    linestyles = ['--'] * len(gal_plotting_l)
    linestyles[0] = '-.'
    linestyles[-1] = '-'

    # fig_masses, ax_masses = plt.subplots(figsize=(10, 6))
    fig_masses, ax_masses = plt.subplots(figsize=(9, 6))

    for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
        # varythis_val=vary_vals[varythis]

        vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

        ax_masses.plot(lookbacktime_l[2:], halo_plot['mtot'][2:], label=f"mtot ({vary_str})", c="xkcd:red", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mgas'][2:], label=f"mgas ({vary_str})", c="xkcd:blue", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mstar'][2:], label=f"mstar ({vary_str})", c="xkcd:orange", ls=ls)
        ax_masses.plot(lookbacktime_l[2:], gal_plot['mout'][2:], label=f"mout ({vary_str})", c="xkcd:green", ls=ls)

    # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
    # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
    # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
    # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
    # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

    # ax_masses_twin = ax_masses.twiny()
    # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

    ax_masses.invert_xaxis()

    ax_masses.set_xlabel('Lookbacktime (Gyr)')
    ax_masses.set_ylabel(r'Masses (M$_\odot$)')
    ax_masses.set_yscale('log')

    # # out legend outside the axes
    # box = ax_masses.get_position()
    # ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig_masses.show()


    # fig_rates, ax_rates = plt.subplots(figsize=(10,6))
    fig_rates, ax_rates = plt.subplots(figsize=(9,6))

    for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
        # varythis_val=vary_vals[varythis]

        vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

        ax_rates.plot(lookbacktime_l[2:], halo_plot['MIR'][2:], label=f"MIR ({vary_str})", c="xkcd:red", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['GCR'][2:], label=f"GCR ({vary_str})", c="xkcd:blue", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['SFR'][2:], label=f"SFR ({vary_str})", c="xkcd:orange", ls=ls)
        ax_rates.plot(lookbacktime_l[2:], gal_plot['MLR'][2:], label=f"MLR ({vary_str})", c="xkcd:green", ls=ls)


    ax_rates.invert_xaxis()

    ax_rates.set_xlabel('Lookbacktime (Gyr)')
    ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
    ax_rates.set_yscale('log')

    # # put legend outside the axes
    # box = ax_rates.get_position()
    # ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig_rates.show()

    return


def test_multiple_evolution_timedirection_andback(
        with_burnin = False,
        zstart: float = 2.,
        zend: float = 0.,
        timestep: float = 1.e-3,
        vary_params = None,
        write_todisk: int = 10,
        directions: tuple = (
                {'name': 'forward', 'factor': 1},
                {'name': 'backward', 'factor': -1}
        ),
        silent: bool = False):

    # subdirs = [itm['name'] + '/' for itm in directions]

    vary_params = {
        'fgal': (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        'SFE': (0.01, 0.05, 0.10, 0.15, 0.20),
        'HLF': (0.01, 0.05, 0.10, 0.20)
    } if vary_params is None else vary_params

    # calc. lookback time and number of steps from zstart and timestep
    lookbacktime = cosmo.lookback_time(zstart).value
    endtime = cosmo.lookback_time(zend).value
    if timestep != 0:
        n_steps = int(np.floor((lookbacktime - endtime) / (directions[0]['factor'] * timestep)))    # works for both directions in time!
    else:
        n_steps = 1     # WARNING: this is a trivial case with just one timestep running
        print("WARNING: zstart = zend, so only one timestep will be cpmputed!")
    # n_steps_maxlen = len(str(n_steps))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'running until z={zend:.3}')
    print(f'corresponding lookback time is {endtime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    # ------------------------------------------
    # make grid with possible value combinations
    vary_n = len(vary_params)
    vary_list = [v for v in vary_params.values()]

    # generate empty array of dimensions to fit the n params in the innermost part (in all j1*j2*...jn combinations)
    vary_grid_dims = tuple(len(vals) for vals in vary_list + [vary_list])
    vary_grid = np.full(shape=vary_grid_dims, fill_value=np.nan)

    for i, vals in enumerate(vary_list):
        # pull one axis after the other to the forefront (rollaxis returns just a view with axis i as axis 0)
        # last dimension has length vary_n and keeps one specific parameter combination (e.g. mstar=10**10, mgas=...,)
        _grid_view = np.rollaxis(vary_grid, i)

        # iterate over the different possible values for this vals (a slice from vary_param)
        # and deposit one value of param 0 at index i=0 in the last dim, one value of param 1 at index i=1, and so on...
        for grid_line, v in zip(_grid_view, vals):
            grid_line[..., i] = v

    # ---------------------------------------
    # create environment, halos and galaxies based on the vary_grid array

    # set up arrays for Halos and Galaxies
    obj_grid_dims = tuple(len(vals) for vals in vary_list)
    halo_grid = np.empty(shape=obj_grid_dims, dtype=object)
    galaxy_grid = np.empty_like(halo_grid, dtype=object)

    # create flat views on the object arrays and for the parameter-tuples array
    vary_grid_flat = vary_grid.reshape(-1, vary_grid.shape[-1])
    # halo_grid_flat = halo_grid.flat
    # galaxy_grid_flat = galaxy_grid.flat

    # create environment
    env = sgm.Environment(name='Environment', zstart=zstart)

    # create halos and corresponding galaxies
    # reference for values in last dim of vary_grid / vary_grid_flat:   (BECAUSE STILL HARDCODED!!!)
    #   0: fgal     (Galaxy)
    #   1: SFE      (Galaxy)
    #   2: MLF      (Galaxy)
    #   3: HLF      (Halo)
    #   4: mstar    (Galaxy)
    for i, valset in enumerate(vary_grid_flat):
        # ASSIGNMENT of vary-values IS STILL HARD-CODED!!!!!!!    (in part due to identical name in Halo and Galaxy)
        halo_grid.flat[i] = env.create_Halo(
            name= f'Halo_{i}',
            mgas= 10**10,
            mdm= 10**12.5,
            HLF= valset[3]
        )
        galaxy_grid.flat[i] = halo_grid.flat[i].create_Galaxy(
            name= f'Galaxy_{i}',
            mgas= valset[4]/10,
            mstar= valset[4],
            fgal=valset[0],
            SFE=valset[1],
            MLF=valset[2]
        )


    if not silent:
        print(env)
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            print(halo)
            print(gal)


    for direction in directions:

        # if not silent:
        #     print(f"Running the simulation {direction['name']}...")
        print(f"Running the simulation {direction['name']}...")


        halo_history_l = []*len(halo_grid.flat)
        for halo in halo_grid.flat:   # make first snapshot (capturing ICs)
            snp = halo.make_snapshot()
            halo_history_l.append([snp.data])
            # writing to disk
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')

        gal_history_l = []*len(galaxy_grid.flat)
        for gal in galaxy_grid.flat:   # make first snapshot (capturing ICs)
            snp = gal.make_snapshot()
            gal_history_l.append([snp.data])
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')


        if not silent:
            print(f"{0}, z={env.z}, lookbacktime={env.lookbacktime}")

        for t in trange(1, n_steps+1):  # from lookbacktime at z=2, only 1/10 of prev. resolution
            # for t in range(12857):  # from lookbacktime at z=2
            # for t in range(101):  # from lookbacktime at z=6
            env.evolve(timestep=direction['factor']*timestep, mode='intuitive')


            for halo, halo_hist in zip(halo_grid.flat, halo_history_l):   # make snapshot of each step
                snp = halo.make_snapshot()
                halo_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            for gal, gal_hist in zip(galaxy_grid.flat, gal_history_l):   # make first snapshot (capturing ICs)
                snp = gal.make_snapshot()
                gal_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            if not silent:
                print(f"{t+1}, z={env.z}, lookbacktime={env.lookbacktime}")

        # if not silent:
        env_final_fp = env.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{env.name}_final.json')
        print(env_final_fp)

        halo_final_fp_l = []
        gal_final_fp_l = []
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            halo_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{halo.name}_final.json'))
            print(halo_final_fp_l[-1])

            gal_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{gal.name}_final.json'))
            print(gal_final_fp_l[-1])



        # ======================
        # grab data for plotting


        # timesteps list for plotting
        timesteps_l = np.linspace(0, lookbacktime, n_steps)

        # grab the data we want to plot from history of galaxy
        lookbacktime_l = [gal['lookbacktime'] for gal in gal_history_l[0]]
        z_l = [gal['z'] for gal in gal_history_l[0]]

        halo_plotting_l = [] * len(halo_history_l)
        gal_plotting_l = [] * len(gal_history_l)
        for (halo_hist, gal_hist) in zip(halo_history_l, gal_history_l):
            gal_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                                   'mstar': [gal['mstar'] for gal in gal_hist],
                                   'mout': [gal['mout'] for gal in gal_hist],
                                   'GCR': [gal['GCR'] for gal in gal_hist],
                                   'SFR': [gal['SFR'] for gal in gal_hist],
                                   'MLR': [gal['MLR'] for gal in gal_hist]})
            halo_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                    'MIR': [halo['MIR'] for halo in halo_hist]})

        # ========
        # plotting

        # # expand colour cycle
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

        linestyles = ['--'] * len(gal_plotting_l)
        linestyles[0] = '-.'
        linestyles[-1] = '-'

        # fig_masses, ax_masses = plt.subplots(figsize=(10, 6))
        fig_masses, ax_masses = plt.subplots(figsize=(9, 6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_masses.plot(lookbacktime_l[2:-2], halo_plot['mtot'][2:-2], label=f"mtot ({vary_str})", c="xkcd:red", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mgas'][2:-2], label=f"mgas ({vary_str})", c="xkcd:blue", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mstar'][2:-2], label=f"mstar ({vary_str})", c="xkcd:orange", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mout'][2:-2], label=f"mout ({vary_str})", c="xkcd:green", ls=ls)

        # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
        # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
        # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
        # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

        # ax_masses_twin = ax_masses.twiny()
        # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

        ax_masses.invert_xaxis()

        ax_masses.set_xlabel('Lookbacktime (Gyr)')
        ax_masses.set_ylabel(r'Masses (M$_\odot$)')
        ax_masses.set_yscale('log')

        # # out legend outside the axes
        # box = ax_masses.get_position()
        # ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_masses.show()

        fig_masses.savefig(f'plots/_tmp/masses_{direction["name"]}.png')


        # fig_rates, ax_rates = plt.subplots(figsize=(10,6))
        fig_rates, ax_rates = plt.subplots(figsize=(9,6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_rates.plot(lookbacktime_l[2:-2], halo_plot['MIR'][2:-2], label=f"MIR ({vary_str})", c="xkcd:red", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['GCR'][2:-2], label=f"GCR ({vary_str})", c="xkcd:blue", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['SFR'][2:-2], label=f"SFR ({vary_str})", c="xkcd:orange", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['MLR'][2:-2], label=f"MLR ({vary_str})", c="xkcd:green", ls=ls)


        ax_rates.invert_xaxis()

        ax_rates.set_xlabel('Lookbacktime (Gyr)')
        ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
        ax_rates.set_yscale('log')

        # # put legend outside the axes
        # box = ax_rates.get_position()
        # ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_rates.show()

        fig_rates.savefig(f'plots/_tmp/mass_rates_{direction["name"]}.png')



        # -----------------------------------------
        # the big switch to run it all backward now         # hardcoded backwards/forwards or other way around

        lookbacktime, endtime = endtime, lookbacktime




    #
    # # ===========================================================================
    # # B A C K W A R D S   P A R T   S T A R T S   H E R E   - RE-USE LATEST STATE
    # # ===========================================================================
    #
    #
    # print('Additional backwards run starts now.')
    #





    return



def test_multiple_evolution_timedirection_andback_haloscaling(
        with_burnin = False,
        zstart: float = 2.,
        zend: float = 0.,
        timestep: float = 1.e-3,
        vary_params = None,
        write_todisk: int = 10,
        directions: tuple = (
                {'name': 'forward', 'factor': 1},
                {'name': 'backward', 'factor': -1}
        ),
        silent: bool = False):

    # subdirs = [itm['name'] + '/' for itm in directions]

    vary_params = {
        'fgal': (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        'SFE': (0.01, 0.05, 0.10, 0.15, 0.20),
        'HLF': (0.01, 0.05, 0.10, 0.20)
    } if vary_params is None else vary_params

    # calc. lookback time and number of steps from zstart and timestep
    lookbacktime = cosmo.lookback_time(zstart).value
    endtime = cosmo.lookback_time(zend).value
    if timestep != 0:
        n_steps = int(np.floor((lookbacktime - endtime) / (directions[0]['factor'] * timestep)))    # works for both directions in time!
    else:
        n_steps = 1     # WARNING: this is a trivial case with just one timestep running
        print("WARNING: zstart = zend, so only one timestep will be cpmputed!")
    # n_steps_maxlen = len(str(n_steps))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'running until z={zend:.3}')
    print(f'corresponding lookback time is {endtime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    # ------------------------------------------
    # make grid with possible value combinations
    vary_n = len(vary_params)
    vary_list = [v for v in vary_params.values()]

    # generate empty array of dimensions to fit the n params in the innermost part (in all j1*j2*...jn combinations)
    vary_grid_dims = tuple(len(vals) for vals in vary_list + [vary_list])
    vary_grid = np.full(shape=vary_grid_dims, fill_value=np.nan)

    for i, vals in enumerate(vary_list):
        # pull one axis after the other to the forefront (rollaxis returns just a view with axis i as axis 0)
        # last dimension has length vary_n and keeps one specific parameter combination (e.g. mstar=10**10, mgas=...,)
        _grid_view = np.rollaxis(vary_grid, i)

        # iterate over the different possible values for this vals (a slice from vary_param)
        # and deposit one value of param 0 at index i=0 in the last dim, one value of param 1 at index i=1, and so on...
        for grid_line, v in zip(_grid_view, vals):
            grid_line[..., i] = v

    # ---------------------------------------
    # create environment, halos and galaxies based on the vary_grid array

    # set up arrays for Halos and Galaxies
    obj_grid_dims = tuple(len(vals) for vals in vary_list)
    halo_grid = np.empty(shape=obj_grid_dims, dtype=object)
    galaxy_grid = np.empty_like(halo_grid, dtype=object)

    # create flat views on the object arrays and for the parameter-tuples array
    vary_grid_flat = vary_grid.reshape(-1, vary_grid.shape[-1])
    # halo_grid_flat = halo_grid.flat
    # galaxy_grid_flat = galaxy_grid.flat

    # create environment
    env = sgm.Environment(name='Environment', zstart=zstart)

    # create halos and corresponding galaxies
    # reference for values in last dim of vary_grid / vary_grid_flat:   (BECAUSE STILL HARDCODED!!!)
    #   0: fgal     (Galaxy)
    #   1: SFE      (Galaxy)
    #   2: MLF      (Galaxy)
    #   3: HLF      (Halo)
    #   4: mstar    (Galaxy)
    #   5: mgas     (Galaxy)
    # also, mhalo will be calc. acc. to Moster+2010, Eq.2 and Table 1, and distrib. 85% to mdm and 15% to mgas of Halo
    for i, valset in enumerate(vary_grid_flat):
        # ASSIGNMENT of vary-values IS STILL HARD-CODED!!!!!!!    (in part due to identical name in Halo and Galaxy)
        _mhalo = iter_mhalo_from_mstar(valset[4])
        halo_grid.flat[i] = env.create_Halo(
            name= f'Halo_{i}',
            mgas= 0.15*_mhalo,
            mdm= 0.85*_mhalo,
            mtot=_mhalo,
            HLF= valset[3]
        )
        galaxy_grid.flat[i] = halo_grid.flat[i].create_Galaxy(
            name= f'Galaxy_{i}',
            mgas= valset[5],
            mstar= valset[4],
            fgal=valset[0],
            SFE=valset[1],
            MLF=valset[2]
        )


    if not silent:
        print(env)
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            print(halo)
            print(gal)


    for direction in directions:

        # if not silent:
        #     print(f"Running the simulation {direction['name']}...")
        print(f"Running the simulation {direction['name']}...")


        halo_history_l = []*len(halo_grid.flat)
        for halo in halo_grid.flat:   # make first snapshot (capturing ICs)
            snp = halo.make_snapshot()
            halo_history_l.append([snp.data])
            # writing to disk
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')

        gal_history_l = []*len(galaxy_grid.flat)
        for gal in galaxy_grid.flat:   # make first snapshot (capturing ICs)
            snp = gal.make_snapshot()
            gal_history_l.append([snp.data])
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')


        if not silent:
            print(f"{0}, z={env.z}, lookbacktime={env.lookbacktime}")

        for t in trange(1, n_steps+1):  # from lookbacktime at z=2, only 1/10 of prev. resolution
            # for t in range(12857):  # from lookbacktime at z=2
            # for t in range(101):  # from lookbacktime at z=6
            env.evolve(timestep=direction['factor']*timestep, mode='intuitive')


            for halo, halo_hist in zip(halo_grid.flat, halo_history_l):   # make snapshot of each step
                snp = halo.make_snapshot()
                halo_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            for gal, gal_hist in zip(galaxy_grid.flat, gal_history_l):   # make first snapshot (capturing ICs)
                snp = gal.make_snapshot()
                gal_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            if not silent:
                print(f"{t+1}, z={env.z}, lookbacktime={env.lookbacktime}")

        # if not silent:
        env_final_fp = env.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{env.name}_final.json')
        print(env_final_fp)

        halo_final_fp_l = []
        gal_final_fp_l = []
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            halo_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{halo.name}_final.json'))
            print(halo_final_fp_l[-1])

            gal_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{gal.name}_final.json'))
            print(gal_final_fp_l[-1])



        # ======================
        # grab data for plotting


        # timesteps list for plotting
        timesteps_l = np.linspace(0, lookbacktime, n_steps)

        # grab the data we want to plot from history of galaxy
        lookbacktime_l = [gal['lookbacktime'] for gal in gal_history_l[0]]
        z_l = [gal['z'] for gal in gal_history_l[0]]

        halo_plotting_l = [] * len(halo_history_l)
        gal_plotting_l = [] * len(gal_history_l)
        for (halo_hist, gal_hist) in zip(halo_history_l, gal_history_l):
            gal_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                                   'mstar': [gal['mstar'] for gal in gal_hist],
                                   'mout': [gal['mout'] for gal in gal_hist],
                                   'GCR': [gal['GCR'] for gal in gal_hist],
                                   'SFR': [gal['SFR'] for gal in gal_hist],
                                   'MLR': [gal['MLR'] for gal in gal_hist]})
            halo_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                    'MIR': [halo['MIR'] for halo in halo_hist]})

        # ========
        # plotting

        # # expand colour cycle
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

        linestyles = ['--'] * len(gal_plotting_l)
        linestyles[0] = '-.'
        linestyles[-1] = '-'

        # fig_masses, ax_masses = plt.subplots(figsize=(10, 6))
        fig_masses, ax_masses = plt.subplots(figsize=(9, 6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_masses.plot(lookbacktime_l[2:-2], halo_plot['mtot'][2:-2], label=f"mtot ({vary_str})", c="xkcd:red", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mgas'][2:-2], label=f"mgas ({vary_str})", c="xkcd:blue", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mstar'][2:-2], label=f"mstar ({vary_str})", c="xkcd:orange", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mout'][2:-2], label=f"mout ({vary_str})", c="xkcd:green", ls=ls)

        # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
        # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
        # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
        # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

        # ax_masses_twin = ax_masses.twiny()
        # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

        ax_masses.invert_xaxis()

        ax_masses.set_xlabel('Lookbacktime (Gyr)')
        ax_masses.set_ylabel(r'Masses (M$_\odot$)')
        ax_masses.set_yscale('log')

        # # out legend outside the axes
        # box = ax_masses.get_position()
        # ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_masses.show()

        fig_masses.savefig(f'plots/_tmp/masses_{direction["name"]}.png')


        # fig_rates, ax_rates = plt.subplots(figsize=(10,6))
        fig_rates, ax_rates = plt.subplots(figsize=(9,6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_rates.plot(lookbacktime_l[2:-2], halo_plot['MIR'][2:-2], label=f"MIR ({vary_str})", c="xkcd:red", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['GCR'][2:-2], label=f"GCR ({vary_str})", c="xkcd:blue", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['SFR'][2:-2], label=f"SFR ({vary_str})", c="xkcd:orange", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['MLR'][2:-2], label=f"MLR ({vary_str})", c="xkcd:green", ls=ls)


        ax_rates.invert_xaxis()

        ax_rates.set_xlabel('Lookbacktime (Gyr)')
        ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
        ax_rates.set_yscale('log')

        # # put legend outside the axes
        # box = ax_rates.get_position()
        # ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_rates.show()

        fig_rates.savefig(f'plots/_tmp/mass_rates_{direction["name"]}.png')



        # -----------------------------------------
        # the big switch to run it all backward now         # hardcoded backwards/forwards or other way around

        lookbacktime, endtime = endtime, lookbacktime




    #
    # # ===========================================================================
    # # B A C K W A R D S   P A R T   S T A R T S   H E R E   - RE-USE LATEST STATE
    # # ===========================================================================
    #
    #
    # print('Additional backwards run starts now.')
    #





    return



def test_multiple_evolution_timedirection_andback_haloscaling_mgasfromSFR(
        with_burnin = False,
        zstart: float = 2.,
        zend: float = 0.,
        timestep: float = 1.e-3,
        vary_params = None,
        write_todisk: int = 10,
        directions: tuple = (
                {'name': 'forward', 'factor': 1},
                {'name': 'backward', 'factor': -1}
        ),
        silent: bool = False):

    # subdirs = [itm['name'] + '/' for itm in directions]

    vary_params = {
        'fgal': (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        'SFE': (0.01, 0.05, 0.10, 0.15, 0.20),
        'HLF': (0.01, 0.05, 0.10, 0.20)
    } if vary_params is None else vary_params

    # calc. lookback time and number of steps from zstart and timestep
    lookbacktime = cosmo.lookback_time(zstart).value
    endtime = cosmo.lookback_time(zend).value
    if timestep != 0:
        n_steps = int(np.floor((lookbacktime - endtime) / (directions[0]['factor'] * timestep)))    # works for both directions in time!
    else:
        n_steps = 1     # WARNING: this is a trivial case with just one timestep running
        print("WARNING: zstart = zend, so only one timestep will be cpmputed!")
    # n_steps_maxlen = len(str(n_steps))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'running until z={zend:.3}')
    print(f'corresponding lookback time is {endtime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    # ------------------------------------------
    # make grid with possible value combinations
    vary_n = len(vary_params)
    vary_list = [v for v in vary_params.values()]

    # generate empty array of dimensions to fit the n params in the innermost part (in all j1*j2*...jn combinations)
    vary_grid_dims = tuple(len(vals) for vals in vary_list + [vary_list])
    vary_grid = np.full(shape=vary_grid_dims, fill_value=np.nan)

    for i, vals in enumerate(vary_list):
        # pull one axis after the other to the forefront (rollaxis returns just a view with axis i as axis 0)
        # last dimension has length vary_n and keeps one specific parameter combination (e.g. mstar=10**10, mgas=...,)
        _grid_view = np.rollaxis(vary_grid, i)

        # iterate over the different possible values for this vals (a slice from vary_param)
        # and deposit one value of param 0 at index i=0 in the last dim, one value of param 1 at index i=1, and so on...
        for grid_line, v in zip(_grid_view, vals):
            grid_line[..., i] = v

    # ---------------------------------------
    # create environment, halos and galaxies based on the vary_grid array

    # set up arrays for Halos and Galaxies
    obj_grid_dims = tuple(len(vals) for vals in vary_list)
    halo_grid = np.empty(shape=obj_grid_dims, dtype=object)
    galaxy_grid = np.empty_like(halo_grid, dtype=object)

    # create flat views on the object arrays and for the parameter-tuples array
    vary_grid_flat = vary_grid.reshape(-1, vary_grid.shape[-1])
    # halo_grid_flat = halo_grid.flat
    # galaxy_grid_flat = galaxy_grid.flat

    # create environment
    env = sgm.Environment(name='Environment', zstart=zstart)

    # create halos and corresponding galaxies
    # reference for values in last dim of vary_grid / vary_grid_flat:   (BECAUSE STILL HARDCODED!!!)
    #   0: fgal     (Galaxy)
    #   1: SFE      (Galaxy)
    #   2: MLF      (Galaxy)
    #   3: HLF      (Halo)
    #   4: mstar    (Galaxy)
    #   5: mgas     (Galaxy)
    # also, mhalo will be calc. acc. to Moster+2010, Eq.2 and Table 1, and distrib. 85% to mdm and 15% to mgas of Halo
    for i, valset in enumerate(vary_grid_flat):
        # ASSIGNMENT of vary-values IS STILL HARD-CODED!!!!!!!    (in part due to identical name in Halo and Galaxy)
        _mhalo = iter_mhalo_from_mstar(valset[4])
        halo_grid.flat[i] = env.create_Halo(
            name= f'Halo_{i}',
            mgas= 0.15*_mhalo,
            mdm= 0.85*_mhalo,
            mtot=_mhalo,
            HLF= valset[3]
        )
        galaxy_grid.flat[i] = halo_grid.flat[i].create_Galaxy(
            name= f'Galaxy_{i}',
            mgas= valset[5]/valset[1],
            mstar= valset[4],
            SFR=valset[5],
            fgal=valset[0],
            SFE=valset[1],
            MLF=valset[2]
        )


    if not silent:
        print(env)
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            print(halo)
            print(gal)


    for direction in directions:

        # if not silent:
        #     print(f"Running the simulation {direction['name']}...")
        print(f"Running the simulation {direction['name']}...")


        halo_history_l = []*len(halo_grid.flat)
        for halo in halo_grid.flat:   # make first snapshot (capturing ICs)
            snp = halo.make_snapshot()
            halo_history_l.append([snp.data])
            # writing to disk
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')

        gal_history_l = []*len(galaxy_grid.flat)
        for gal in galaxy_grid.flat:   # make first snapshot (capturing ICs)
            snp = gal.make_snapshot()
            gal_history_l.append([snp.data])
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')


        if not silent:
            print(f"{0}, z={env.z}, lookbacktime={env.lookbacktime}")

        for t in trange(1, n_steps+1):  # from lookbacktime at z=2, only 1/10 of prev. resolution
            # for t in range(12857):  # from lookbacktime at z=2
            # for t in range(101):  # from lookbacktime at z=6
            env.evolve(timestep=direction['factor']*timestep, mode='intuitive')


            for halo, halo_hist in zip(halo_grid.flat, halo_history_l):   # make snapshot of each step
                snp = halo.make_snapshot()
                halo_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            for gal, gal_hist in zip(galaxy_grid.flat, gal_history_l):   # make first snapshot (capturing ICs)
                snp = gal.make_snapshot()
                gal_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            if not silent:
                print(f"{t+1}, z={env.z}, lookbacktime={env.lookbacktime}")

        # if not silent:
        env_final_fp = env.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{env.name}_final.json')
        print(env_final_fp)

        halo_final_fp_l = []
        gal_final_fp_l = []
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            halo_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{halo.name}_final.json'))
            print(halo_final_fp_l[-1])

            gal_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{gal.name}_final.json'))
            print(gal_final_fp_l[-1])



        # ======================
        # grab data for plotting


        # timesteps list for plotting
        timesteps_l = np.linspace(0, lookbacktime, n_steps)

        # grab the data we want to plot from history of galaxy
        lookbacktime_l = [gal['lookbacktime'] for gal in gal_history_l[0]]
        z_l = [gal['z'] for gal in gal_history_l[0]]

        halo_plotting_l = [] * len(halo_history_l)
        gal_plotting_l = [] * len(gal_history_l)
        for (halo_hist, gal_hist) in zip(halo_history_l, gal_history_l):
            gal_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                                   'mstar': [gal['mstar'] for gal in gal_hist],
                                   'mout': [gal['mout'] for gal in gal_hist],
                                   'GCR': [gal['GCR'] for gal in gal_hist],
                                   'SFR': [gal['SFR'] for gal in gal_hist],
                                   'MLR': [gal['MLR'] for gal in gal_hist]})
            halo_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                    'MIR': [halo['MIR'] for halo in halo_hist]})

        # ========
        # plotting

        # # expand colour cycle
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

        linestyles = ['--'] * len(gal_plotting_l)
        linestyles[0] = '-.'
        linestyles[-1] = '-'

        # fig_masses, ax_masses = plt.subplots(figsize=(10, 6))
        fig_masses, ax_masses = plt.subplots(figsize=(9, 6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_masses.plot(lookbacktime_l[2:-2], halo_plot['mtot'][2:-2], label=f"mtot ({vary_str})", c="xkcd:red", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mgas'][2:-2], label=f"mgas ({vary_str})", c="xkcd:blue", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mstar'][2:-2], label=f"mstar ({vary_str})", c="xkcd:orange", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mout'][2:-2], label=f"mout ({vary_str})", c="xkcd:green", ls=ls)

        # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
        # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
        # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
        # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

        # ax_masses_twin = ax_masses.twiny()
        # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

        ax_masses.invert_xaxis()

        ax_masses.set_xlabel('Lookbacktime (Gyr)')
        ax_masses.set_ylabel(r'Masses (M$_\odot$)')
        ax_masses.set_yscale('log')

        # # out legend outside the axes
        # box = ax_masses.get_position()
        # ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_masses.show()

        fig_masses.savefig(f'plots/_tmp/masses_{direction["name"]}.png')


        # fig_rates, ax_rates = plt.subplots(figsize=(10,6))
        fig_rates, ax_rates = plt.subplots(figsize=(9,6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_rates.plot(lookbacktime_l[2:-2], halo_plot['MIR'][2:-2], label=f"MIR ({vary_str})", c="xkcd:red", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['GCR'][2:-2], label=f"GCR ({vary_str})", c="xkcd:blue", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['SFR'][2:-2], label=f"SFR ({vary_str})", c="xkcd:orange", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['MLR'][2:-2], label=f"MLR ({vary_str})", c="xkcd:green", ls=ls)


        ax_rates.invert_xaxis()

        ax_rates.set_xlabel('Lookbacktime (Gyr)')
        ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
        ax_rates.set_yscale('log')

        # # put legend outside the axes
        # box = ax_rates.get_position()
        # ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_rates.show()

        fig_rates.savefig(f'plots/_tmp/mass_rates_{direction["name"]}.png')



        # -----------------------------------------
        # the big switch to run it all backward now         # hardcoded backwards/forwards or other way around

        lookbacktime, endtime = endtime, lookbacktime




    #
    # # ===========================================================================
    # # B A C K W A R D S   P A R T   S T A R T S   H E R E   - RE-USE LATEST STATE
    # # ===========================================================================
    #
    #
    # print('Additional backwards run starts now.')
    #





    return



def test_multiple_evolution_timedirection_andback_haloscaling_paramarray(
        with_burnin = False,
        zstart: float = 2.,
        zend: float = 0.,
        timestep: float = 1.e-3,
        vary_params = None,
        write_todisk: int = 10,
        directions: tuple = (
                {'name': 'forward', 'factor': 1},
                {'name': 'backward', 'factor': -1}
        ),
        silent: bool = False):

    # subdirs = [itm['name'] + '/' for itm in directions]

    vary_params = {
        'fgal': (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        'SFE': (0.01, 0.05, 0.10, 0.15, 0.20),
        'HLF': (0.01, 0.05, 0.10, 0.20)
    } if vary_params is None else vary_params

    # calc. lookback time and number of steps from zstart and timestep
    lookbacktime = cosmo.lookback_time(zstart).value
    endtime = cosmo.lookback_time(zend).value
    if timestep != 0:
        n_steps = int(np.floor((lookbacktime - endtime) / (directions[0]['factor'] * timestep)))    # works for both directions in time!
    else:
        n_steps = 1     # WARNING: this is a trivial case with just one timestep running
        print("WARNING: zstart = zend, so only one timestep will be cpmputed!")
    # n_steps_maxlen = len(str(n_steps))

    print(f'starting from z={zstart:.3}')
    print(f'corresponding lookback time is {lookbacktime:.4} Gyrs')
    print(f'running until z={zend:.3}')
    print(f'corresponding lookback time is {endtime:.4} Gyrs')
    print(f'timestep width is {timestep:.6} Gyrs')
    print(f'resulting in {n_steps} timesteps')

    # ------------------------------------------
    # make grid with possible value combinations
    vary_n = len(vary_params)
    vary_list = [v for v in vary_params.values()]

    # generate empty array of dimensions to fit the n params in the innermost part (in all j1*j2*...jn combinations)
    vary_grid_dims = tuple(len(vals) for vals in vary_list + [vary_list])
    vary_grid = np.full(shape=vary_grid_dims, fill_value=np.nan)

    for i, vals in enumerate(vary_list):
        # pull one axis after the other to the forefront (rollaxis returns just a view with axis i as axis 0)
        # last dimension has length vary_n and keeps one specific parameter combination (e.g. mstar=10**10, mgas=...,)
        _grid_view = np.rollaxis(vary_grid, i)

        # iterate over the different possible values for this vals (a slice from vary_param)
        # and deposit one value of param 0 at index i=0 in the last dim, one value of param 1 at index i=1, and so on...
        for grid_line, v in zip(_grid_view, vals):
            grid_line[..., i] = v

    # ---------------------------------------
    # create environment, halos and galaxies based on the vary_grid array

    # set up arrays for Halos and Galaxies
    obj_grid_dims = tuple(len(vals) for vals in vary_list)
    halo_grid = np.empty(shape=obj_grid_dims, dtype=object)
    galaxy_grid = np.empty_like(halo_grid, dtype=object)

    # create flat views on the object arrays and for the parameter-tuples array
    vary_grid_flat = vary_grid.reshape(-1, vary_grid.shape[-1])
    # halo_grid_flat = halo_grid.flat
    # galaxy_grid_flat = galaxy_grid.flat

    # create environment
    env = sgm.Environment(name='Environment', zstart=zstart)

    # create halos and corresponding galaxies
    # reference for values in last dim of vary_grid / vary_grid_flat:   (BECAUSE STILL HARDCODED!!!)
    #   0: fgal     (Galaxy)
    #   1: SFE      (Galaxy)
    #   2: MLF      (Galaxy)
    #   3: HLF      (Halo)
    #   4: mstar    (Galaxy)
    #   5: mgas     (Galaxy)
    # also, mhalo will be calc. acc. to Moster+2010, Eq.2 and Table 1, and distrib. 85% to mdm and 15% to mgas of Halo
    for i, valset in enumerate(vary_grid_flat):
        # ASSIGNMENT of vary-values IS STILL HARD-CODED!!!!!!!    (in part due to identical name in Halo and Galaxy)
        _mhalo = iter_mhalo_from_mstar(valset[4])
        halo_grid.flat[i] = env.create_Halo(
            name= f'Halo_{i}',
            mgas= 0.15*_mhalo,
            mdm= 0.85*_mhalo,
            mtot=_mhalo,
            HLF= valset[3]
        )
        galaxy_grid.flat[i] = halo_grid.flat[i].create_Galaxy(
            name= f'Galaxy_{i}',
            mgas= valset[5],
            mstar= valset[4],
            fgal=valset[0],
            SFE=valset[1],
            MLF=valset[2]
        )


    if not silent:
        print(env)
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            print(halo)
            print(gal)


    for direction in directions:

        # if not silent:
        #     print(f"Running the simulation {direction['name']}...")
        print(f"Running the simulation {direction['name']}...")


        halo_history_l = []*len(halo_grid.flat)
        for halo in halo_grid.flat:   # make first snapshot (capturing ICs)
            snp = halo.make_snapshot()
            halo_history_l.append([snp.data])
            # writing to disk
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')

        gal_history_l = []*len(galaxy_grid.flat)
        for gal in galaxy_grid.flat:   # make first snapshot (capturing ICs)
            snp = gal.make_snapshot()
            gal_history_l.append([snp.data])
            if write_todisk > 0:
                snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(0, n_steps)}')


        if not silent:
            print(f"{0}, z={env.z}, lookbacktime={env.lookbacktime}")

        for t in trange(1, n_steps+1):  # from lookbacktime at z=2, only 1/10 of prev. resolution
            # for t in range(12857):  # from lookbacktime at z=2
            # for t in range(101):  # from lookbacktime at z=6
            env.evolve(timestep=direction['factor']*timestep, mode='intuitive')


            for halo, halo_hist in zip(halo_grid.flat, halo_history_l):   # make snapshot of each step
                snp = halo.make_snapshot()
                halo_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            for gal, gal_hist in zip(galaxy_grid.flat, gal_history_l):   # make first snapshot (capturing ICs)
                snp = gal.make_snapshot()
                gal_hist.append(snp.data)
                if (write_todisk > 0) and ((t % write_todisk == 0) or (t == n_steps)):
                    snp.save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{snp.autoname_with_index(t, n_steps)}')

            if not silent:
                print(f"{t+1}, z={env.z}, lookbacktime={env.lookbacktime}")

        # if not silent:
        env_final_fp = env.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{env.name}_final.json')
        print(env_final_fp)

        halo_final_fp_l = []
        gal_final_fp_l = []
        for halo, gal in zip(halo_grid.flat, galaxy_grid.flat):
            halo_final_fp_l.append(halo.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{halo.name}_final.json'))
            print(halo_final_fp_l[-1])

            gal_final_fp_l.append(gal.make_snapshot().save_to_disk(f'outputs/_tmp/' + direction['name'] + f'/{gal.name}_final.json'))
            print(gal_final_fp_l[-1])



        # ======================
        # grab data for plotting


        # timesteps list for plotting
        timesteps_l = np.linspace(0, lookbacktime, n_steps)

        # grab the data we want to plot from history of galaxy
        lookbacktime_l = [gal['lookbacktime'] for gal in gal_history_l[0]]
        z_l = [gal['z'] for gal in gal_history_l[0]]

        halo_plotting_l = [] * len(halo_history_l)
        gal_plotting_l = [] * len(gal_history_l)
        for (halo_hist, gal_hist) in zip(halo_history_l, gal_history_l):
            gal_plotting_l.append({'mgas': [gal['mgas'] for gal in gal_hist],
                                   'mstar': [gal['mstar'] for gal in gal_hist],
                                   'mout': [gal['mout'] for gal in gal_hist],
                                   'GCR': [gal['GCR'] for gal in gal_hist],
                                   'SFR': [gal['SFR'] for gal in gal_hist],
                                   'MLR': [gal['MLR'] for gal in gal_hist]})
            halo_plotting_l.append({'mtot': [halo['mtot'] for halo in halo_hist],
                                    'MIR': [halo['MIR'] for halo in halo_hist]})

        # ========
        # plotting

        # # expand colour cycle
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

        linestyles = ['--'] * len(gal_plotting_l)
        linestyles[0] = '-.'
        linestyles[-1] = '-'

        # fig_masses, ax_masses = plt.subplots(figsize=(10, 6))
        fig_masses, ax_masses = plt.subplots(figsize=(9, 6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_masses.plot(lookbacktime_l[2:-2], halo_plot['mtot'][2:-2], label=f"mtot ({vary_str})", c="xkcd:red", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mgas'][2:-2], label=f"mgas ({vary_str})", c="xkcd:blue", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mstar'][2:-2], label=f"mstar ({vary_str})", c="xkcd:orange", ls=ls)
            ax_masses.plot(lookbacktime_l[2:-2], gal_plot['mout'][2:-2], label=f"mout ({vary_str})", c="xkcd:green", ls=ls)

        # ax.plot(timesteps_l, GAR_l, label="GAR (gas accretion rate)")
        # ax.plot(timesteps_l[5:], sMIR_l[5:], label="sMIR (specific mass increase rate)")
        # ax.plot(timesteps_l[5:], rsSFR_l[5:], label="rsSFR (reduced specific star formation rate)")
        # ax.plot(timesteps_l[5:], fout_l[5:], label="fout (gas ejected as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fgas_l[5:], label="fgas (gas going into reservoir as fraction of infalling gas)")
        # ax.plot(timesteps_l[5:], fstar_l[5:], label="fstar (gas converted to long-lived stars as fraction of infalling gas)")

        # ax_masses_twin = ax_masses.twiny()
        # ax_masses_twin.set_xticks([6, 5, 4, 3, 2, 1, 0])

        ax_masses.invert_xaxis()

        ax_masses.set_xlabel('Lookbacktime (Gyr)')
        ax_masses.set_ylabel(r'Masses (M$_\odot$)')
        ax_masses.set_yscale('log')

        # # out legend outside the axes
        # box = ax_masses.get_position()
        # ax_masses.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_masses.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_masses.show()

        fig_masses.savefig(f'plots/_tmp/masses_{direction["name"]}.png')


        # fig_rates, ax_rates = plt.subplots(figsize=(10,6))
        fig_rates, ax_rates = plt.subplots(figsize=(9,6))

        for gal_plot, halo_plot, vary_vals, ls in zip(gal_plotting_l, halo_plotting_l, vary_grid_flat, linestyles):
            # varythis_val=vary_vals[varythis]

            vary_str = ", ".join(f"{param_key}={vary_vals[i]:.2}" for i, param_key in enumerate(vary_params.keys()))

            ax_rates.plot(lookbacktime_l[2:-2], halo_plot['MIR'][2:-2], label=f"MIR ({vary_str})", c="xkcd:red", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['GCR'][2:-2], label=f"GCR ({vary_str})", c="xkcd:blue", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['SFR'][2:-2], label=f"SFR ({vary_str})", c="xkcd:orange", ls=ls)
            ax_rates.plot(lookbacktime_l[2:-2], gal_plot['MLR'][2:-2], label=f"MLR ({vary_str})", c="xkcd:green", ls=ls)


        ax_rates.invert_xaxis()

        ax_rates.set_xlabel('Lookbacktime (Gyr)')
        ax_rates.set_ylabel(r'Mass Rates (M$_\odot$ / Gyr)')
        ax_rates.set_yscale('log')

        # # put legend outside the axes
        # box = ax_rates.get_position()
        # ax_rates.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # ax_rates.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig_rates.show()

        fig_rates.savefig(f'plots/_tmp/mass_rates_{direction["name"]}.png')



        # -----------------------------------------
        # the big switch to run it all backward now         # hardcoded backwards/forwards or other way around

        lookbacktime, endtime = endtime, lookbacktime

    return




def main():
    # testrun_1(silent=True)
    # testrun_2(silent=True)
    # testrun_3(silent=True)
    # testrun_4(silent=True)
    # testrun_5(silent=True)
    # test_fgal1(silent=True)
    # test_sfe1(zstart=2., timestep=1.e-2, silent=True)
    # test_multiple(
    #     # zstart=0.1,
    #     zstart=2.,
    #     timestep=1.e-2,
    #     silent=True,
    #     vary_params={
    #         'fgal': (0.01, 0.10, 0.30, 0.50),
    #         'SFE': (0.01, 0.10, 0.20),
    #         'MLF': (0.1, 0.2),
    #         'HLF': (0.01, 0.10)
    #     }
    # )
    # test_multiple(
    #     # zstart=0.1,
    #     zstart=2.,
    #     timestep=1.e-2,
    #     silent=True,
    #     vary_params={
    #         'fgal': (0.005, 0.01, 0.05, 0.1),
    #         'SFE': (0.005, 0.01, 0.015, 0.02),
    #         'MLF': (0.05, 0.2, 0.5),
    #         'HLF': (0.01, 0.5)
    #     }
    # )
    # test_multiple(
    #     # zstart=0.1,
    #     zstart=2.,
    #     timestep=1.e-2,
    #     silent=True,
    #     vary_params={
    #         'fgal': (0.005, 0.01, 0.025, 0.05, 0.1, 0.15),
    #         'SFE': (1.,),
    #         'MLF': (0.05, 0.1, 0.2, 0.5),
    #         'HLF': (0.01, 0.1, 0.5)
    #     }
    # )
    # test_multiple(
    #     # zstart=0.1,
    #     zstart=2.,
    #     timestep=1.e-2,
    #     silent=True,
    #     vary_params={
    #         'fgal': (0.005, 0.01, 0.025, 0.05, 0.1, 0.15),
    #         'SFE': (0.9, 1., 1.2),
    #         'MLF': (0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
    #         'HLF': (0.1,)
    #     }
    # )
    # test_multiple_evolution(
    #     # zstart=0.1,
    #     zstart=2.,
    #     timestep=1.e-2,
    #     silent=True,
    #     write_todisk=10,
    #     vary_params={
    #         'fgal': (0.01, 0.025, 0.05, 0.1, 0.15),
    #         'SFE': (1.,),
    #         'MLF': (0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
    #         'HLF': (0.1,)
    #     }
    # )
    # return
    # # different-mass galaxies
    # test_multiple_evolution(
    #     # zstart=0.1,
    #     zstart=2.,
    #     timestep=1.e-2,
    #     silent=True,
    #     write_todisk=10,
    #     vary_params={
    #         'fgal': (0.01, 0.1, 0.25),
    #         'SFE': (1.,),
    #         'MLF': (0.01, 0.1, 0.25),
    #         'HLF': (0.1,),
    #         'mstar': (10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11, 10**11.5)
    #     }
    # )
    # different-mass galaxies backwards in time
    # test_multiple_evolution_timedirection(
    #     # zstart=0.1,
    #     zstart=0.,
    #     # zend=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     zend=2.,
    #     timestep=-1.e-2,
    #     silent=True,
    #     write_todisk=10,
    #     vary_params={
    #         'fgal': (0.01, 0.1, 0.25),
    #         'SFE': (1.,),
    #         'MLF': (0.01, 0.1, 0.25),
    #         'HLF': (0.1,),
    #         'mstar': (10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11, 10**11.5)
    #     }
    # )
    # test_multiple_evolution_timedirection_andback(
    #     zstart=0.1,
    #     # zstart=2.,
    #     # zend=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     # zend=2.,
    #     zend=0.,
    #     timestep=1.e-2,
    #     silent=True,
    #     # write_todisk=10,
    #     write_todisk=1,
    #     vary_params={
    #         'fgal': (0.01, 0.1, 0.25),
    #         'SFE': (1.,),
    #         'MLF': (0.01, 0.1, 0.25),
    #         'HLF': (0.1,),
    #         'mstar': (10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11, 10**11.5)
    #     },
    #     directions=(
    #     {'name': 'forward', 'factor': 1},
    #     {'name': 'backward', 'factor': -1}
    #     )
    # )
    # test_multiple_evolution_timedirection_andback(
    #     zstart=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     # zstart=0.1,
    #     # zstart=2.,
    #     # zend=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     # zend=2.,
    #     zend=0.,
    #     timestep=1.e-3,
    #     silent=True,
    #     # write_todisk=10,
    #     write_todisk=1,
    #     vary_params={
    #         'fgal': (0.01, 0.1, 0.25),
    #         'SFE': (1.,),
    #         'MLF': (0.01, 0.1, 0.25),
    #         'HLF': (0.1,),
    #         'mstar': (10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11, 10**11.5)
    #     },
    #     directions=(
    #     {'name': 'forward', 'factor': 1},
    #     {'name': 'backward', 'factor': -1}
    #     )
    # # )
    # test_multiple_evolution_timedirection_andback_haloscaling(
    #     zstart=0.,
    #     zend=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     timestep=1.e-3,
    #     silent=True,
    #     # write_todisk=10,
    #     write_todisk=1,
    #     vary_params={
    #         'fgal': (0.01, 0.1, 0.25),
    #         'SFE': (1.,),
    #         'MLF': (0.01, 0.1, 0.25),
    #         'HLF': (0.1,),
    #         'mstar': (10**8, 10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11, 10**11.5, 10**12),
    #         'mgas': (10**6.5, 10**7, 10**7.5, 10**8, 10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11)
    #     },
    #     directions=(
    #     {'name': 'backward', 'factor': -1},
    #     {'name': 'forward', 'factor': 1}
    #     )
    # )
    # test_multiple_evolution_timedirection_andback_haloscaling(
    #     zstart=0.,
    #     zend=z_at_value(cosmo.age, cosmo.age(0) - 0.1 * u.Gyr),
    #     # zend=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     timestep=1.e-2,
    #     # timestep=1.e-3,
    #     silent=True,
    #     # write_todisk=10,
    #     write_todisk=1,
    #     vary_params={
    #         'fgal': (0.1,),
    #         # 'fgal': (0.01, 0.1, 0.25),
    #         'SFE': (1.,),
    #         'MLF': (0.1,),
    #         # 'MLF': (0.01, 0.1, 0.25),
    #         'HLF': (0.1,),
    #         'mstar': (10**8, 10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11, 10**11.5, 10**12),
    #         'mgas': (10**6.5, 10**7, 10**7.5, 10**8, 10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11)
    #     },
    #     directions=(
    #     {'name': 'backward', 'factor': -1},
    #     {'name': 'forward', 'factor': 1}
    #     )
    # )
    # test_multiple_evolution_timedirection_andback_haloscaling(
    #     zstart=0.,
    #     zend=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     timestep=1.e-3,
    #     silent=True,
    #     write_todisk=1,
    #     vary_params={
    #         'fgal': (0.1,),
    #         'SFE': (1.,),
    #         'MLF': (0.1,),
    #         'HLF': (0.1,),
    #         'mstar': (10**8, 10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11, 10**11.5, 10**12),
    #         'mgas': (10**6.5, 10**7, 10**7.5, 10**8, 10**8.5, 10**9, 10**9.5, 10**10, 10**10.5, 10**11)
    #     },
    #     directions=(
    #         {'name': 'backward', 'factor': -1},
    #         {'name': 'forward', 'factor': 1}
    #     )
    # )

    # setting paths
    project_dir = Path.cwd()
    sfr79_dir = project_dir / 'data' / "SFR79_grids"
    date_and_time_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    plot_dir = project_dir / 'plots' / '_tmp' / date_and_time_str
    plot_dir.mkdir()
    out_dir = project_dir / 'outputs' / '_tmp' / date_and_time_str  # only defining the path, not creating yet
    # out_dir.mkdir()  # make the output sub-directory

    # reading observational results
    sfr79_medians = np.loadtxt(str(sfr79_dir / "SFR79_2dhist_medians.txt"))
    mstar_mesh =  np.loadtxt(str(sfr79_dir / "SFR79_2dhist_binedges_mstar_mesh.txt"))
    sfr_mesh = np.loadtxt(str(sfr79_dir / "SFR79_2dhist_binedges_sfr_mesh.txt"))
    n_binned = np.loadtxt(str(sfr79_dir / "SFR79_2dhist_binnumbers.txt"))

    # remove bins with less than n objects in them
    n_binned_min = 40
    sfr79_medians = np.where(n_binned >= n_binned_min, sfr79_medians, np.nan)

    # getting the mstar-SFR bin cells that do have observational results
    mstar_and_SFR = calc_bincentres_where_not_nan(sfr79_medians, mstar_mesh, sfr_mesh)

    # intial values for Galaxy properties
    mstar = 10**mstar_and_SFR[:, 0]
    SFR = 10**mstar_and_SFR[:, 1] * 10**9   # CONVERSION of 'per yr' (obs) to 'per Gyr' (sims)
    sSFR = SFR / mstar
    mgas = calculate_mgas_mstar_from_sSFR(sSFR / 10**9, log_values=False, withscatter=False) * mstar
    # SFE = np.array([1.] * len(mstar))
    SFE = SFR / mgas   # set SFE to one (a.t.m. const) unique value, in harmony with the sSFR relation (through mgas)
    # fgal = np.array([0.1] * len(mstar))
    # fgal = np.array([0.01] * len(mstar))
    # fgal = np.array([0.005] * len(mstar))
    fgal = np.array([0.4] * len(mstar))  # following Lilly+13
    MLF = np.array([0.1] * len(mstar))
    # MLF = np.array([0.05] * len(mstar))
    # MLF = np.array([0.2] * len(mstar))

    # initial values for Halo properties
    mhalo = np.array([iter_mhalo_from_mstar(mstar_i) for mstar_i in mstar])
    # BDR = np.array([0.2] * len(mstar))
    BDR = np.array([0.15] * len(mstar))  # actual value from Lilly+13
    HLF = np.array([0.1] * len(mstar))


    # # OLD INTERMEDIATE PLOTTING (worked fine, just SFR79 and mgas)
    # # intermediate plotting: checking that the ICs (mstar and SFR) are actually matching the obs. input
    # sfr79_range = (-2, 2)
    # cmap = mpl.cm.RdBu
    # norm = mpl.colors.Normalize(vmin=sfr79_range[0], vmax=sfr79_range[1])
    # fig, (ax_obs, ax_cbar, ax_cbar_mgas) = plt.subplots(1, 3,
    #                                                      gridspec_kw={
    #                                                          'width_ratios': (9, 1, 1),
    #                                                          'hspace': 0.05
    #                                                      },
    #                                                      figsize=(13, 9))
    # im_obs = ax_obs.pcolormesh(mstar_mesh, sfr_mesh,
    #                            sfr79_medians,
    #                            cmap=cmap, norm=norm)
    # # ic_sims = ax_obs.plot(np.log10(mstar), np.log10(SFR) - 9., "o")
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #              ax=ax_cbar,
    #              fraction=0.8,
    #              extend='both',
    #              anchor=(0.0, 0.0),
    #              label='log SFR79')
    #
    # mgas_color_range = (np.min(np.log10(mgas)), np.max(np.log10(mgas)))
    # cmap_mgas = mpl.cm.YlGn
    # norm_mgas = mpl.colors.Normalize(vmin=mgas_color_range[0], vmax=mgas_color_range[1])
    # ic_sims_mgas = ax_obs.scatter(x=np.log10(mstar), y=np.log10(SFR) - 9.,
    #                               c=np.log10(mgas), cmap=cmap_mgas, norm=norm_mgas,
    #                               marker="o",
    #                               zorder=10)
    #
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm_mgas, cmap=cmap_mgas),
    #              ax=ax_cbar_mgas,
    #              fraction=0.8,
    #              # extend='both',
    #              anchor=(0.0, 0.0),
    #              label=r'log $M_{\mathrm{H}_2}$ [$M_\odot$]')
    #
    # ax_cbar.remove()
    # ax_cbar_mgas.remove()
    # ax_obs.set_xlabel(r'log $M_\star$ [$M_\odot$]')
    # ax_obs.set_ylabel(r'log SFR [$M_\odot \, yr^{-1}$]')
    # fig.savefig(plot_dir / f'mstar_SFR_obs_vs_ICs_of_sims_{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}.png')
    #



    # intermediate plotting: checking that the ICs (mstar and SFR) are actually matching the obs. input
    fig, ax_obs = plot_initial_conditions(mstar, SFR, mgas, mhalo, mstar_mesh, sfr_mesh, sfr79_medians, n_binned_min,
                                          plot_dir)

    # numbering Halos and Galaxies
    name_i = np.arange(len(mstar))

    # making all names
    name_env = 'Environment'
    name_halos = [f'Halo_{i}' for i in name_i]
    name_gals = [f'Galaxy_{i}' for i in name_i]

    # Galaxy ICs
    IC_gal_mstar = sgm.IC.single_param('mstar', mstar)
    IC_gal_SFR = sgm.IC.single_param('SFR', SFR)
    IC_gal_sSFR = sgm.IC.single_param('sSFR', sSFR)
    IC_gal_SFE = sgm.IC.single_param('SFE', SFE)
    IC_gal_mgas = sgm.IC.single_param('mgas', mgas)
    IC_gal_fgal = sgm.IC.single_param('fgal', fgal)
    IC_gal_MLF = sgm.IC.single_param('MLF', MLF)

    # Halo ICs
    IC_halo_mtot = sgm.IC.single_param('mtot', mhalo)
    IC_halo_BDR = sgm.IC.single_param('BDR', BDR)
    IC_halo_mdm = sgm.IC.single_param('mdm', (1. / (BDR + 1.)) * mhalo)
    IC_halo_mgas = sgm.IC.single_param('mgas', ((BDR / (BDR + 1.)) * mhalo) - (mgas + mstar))
    IC_halo_HLF = sgm.IC.single_param('HLF', HLF)

    # Environment IC
    IC_env_zstart = sgm.IC.single_param('zstart', [0.])  # can also use 'lookbacktime' (Gyrs) instead of zstart
    # IC_env_lookbacktime = sgm.IC.single_param('lookbacktime', [0.8])  # THIS IS FOR FORWARDS, THEN BACKWARDS COMPARISON

    # combine into one IC object for Environment, one for Halos and one for Galaxies
    IC_env_comb = (IC_env_zstart)
    # IC_env_comb = (IC_env_lookbacktime)  # THIS IS FOR FORWARDS, THEN BACKWARDS COMPARISON
    IC_halo_comb = (IC_halo_mtot +
                    IC_halo_BDR +
                    IC_halo_mdm +
                    IC_halo_mgas +
                    IC_halo_HLF)
    IC_gal_comb = (IC_gal_mstar +
                   IC_gal_SFR +
                   IC_gal_sSFR +
                   IC_gal_SFE +
                   IC_gal_mgas +
                   IC_gal_fgal +
                   IC_gal_MLF)

    # create the AstroObjects according to ICs
    n_env, n_halo, n_gal = len(IC_env_comb), len(IC_halo_comb), len(IC_gal_comb)
    print(f"Creating AstroObjects as sepcified: \n"
          f"  {n_env} Environment object{'s' if n_env > 1 else ''} \n"
          f"  {n_halo} Halo object{'s' if n_halo > 1 else ''} \n"
          f"  {n_gal} Galaxy object{'s' if n_gal > 1 else ''}")
    env, halo_arr, gal_arr = sgm.create_AstroObject_Ensemble(name_env, IC_env_comb,
                                                             name_halos, IC_halo_comb,
                                                             name_gals, IC_gal_comb)


    # double-check that we want to really run the whole integrator
    run_sim = input("Proceed with running simulation? [y/n]").casefold()
    if (run_sim == "y".casefold()) or (run_sim == "yes".casefold()):
        out_dir.mkdir()  # actually make the output sub-directory (since we're sure we'll be running the sim now)
        print(f"Created new output directory '{out_dir}'")

        # # create integrator
        # Integrator = sgm.FTI(
        #     env=env,
        #     evolve_method='evolve',
        #     dt=-1.e-3,
        #     t_start=env.lookbacktime,
        #     t_end=0.8
        # )
        #
        # # run the integrator
        # print("Starting integration")
        # Integrator.integrate(
        #     wtd=1,
        #     outdir=out_dir
        # )


        # # THIS IS FOR FORWARDS, THEN BACKWARDS COMPARISON
        #
        # # create FORWARD integrator
        # Integrator = sgm.FTI(
        #     env=env,
        #     evolve_method='evolve',
        #     # dt=1.e-3,
        #     dt=1.e-4,
        #     t_start=env.lookbacktime,
        #     t_end=0.0
        # )
        #
        # # run the FORWARD integrator
        # print("Starting integration")
        # Integrator.integrate(
        #     wtd=1,
        #     outdir=out_dir / "0_forward_800Myr_1e-4",
        #     single_snapshots=False
        # )
        #
        # # create BACKWARD integrator
        # Integrator = sgm.FTI(
        #     env=env,
        #     evolve_method='evolve',
        #     # dt=-1.e-3,
        #     dt=-1.e-4,
        #     t_start=env.lookbacktime,
        #     t_end=0.8
        # )
        #
        # # run the BACKWARD integrator
        # print("Starting integration")
        # Integrator.integrate(
        #     wtd=1,
        #     outdir=out_dir / "1_backward_800Myr_1e-4",
        #     single_snapshots=False
        # )


        # THIS IS JUST BACKWARDS INTEGRATION FROM z=0, BUT FARTHER

        # create BACKWARD integrator
        Integrator = sgm.FTI(
            env=env,
            evolve_method='evolve',
            dt=-1.e-3,
            t_start=env.lookbacktime,
            t_end=2
        )

        # run the BACKWARD integrator
        print("Starting integration")
        Integrator.integrate(
            wtd=1,
            outdir=out_dir / "0_backward_2Gyr_1e-3",
            single_snapshots=False
        )

    elif (run_sim == "n".casefold()) or (run_sim == "no".casefold()):
        print("Integration not started as per user input")

        # Check whether the user want to explore the attributes of the created AstroObjects
        explore_attrs = input("Do you want to explore the attributes of the AstroObjects that you just created? "
                              "[y/n]").casefold()
        if (explore_attrs == "y".casefold()) or (explore_attrs == "yes".casefold()):
            explore_attrs_and_augment_plot(env, mstar_mesh, sfr_mesh, fig, ax_obs, plot_dir)

            # print(mstar_all[mstar_sfr_match])
            # print(sfr_all[mstar_sfr_match])
            # print(int(np.argwhere(mstar_sfr_match)))
            #  W O R K   H E R E
            #
    else:
        print("Invalid user input, simulation aborted")


    # out_dir = project_dir / 'outputs' / '_tmp' / date_and_time_str
    # out_dir.mkdir()  # make the output sub-directory
    #
    # # run the integrator
    # print("Starting integration")
    # Integrator.integrate(
    #     wtd=1,
    #     outdir=out_dir
    # )



    # # final plotting for overview of time evolution of some quantities:
    # # THIS WOULD BE REALLY NICE, BUT I DON'T MAINTAIN A HISTORY IN MEMORY AT THE MOMENT,
    # # SO IT'S IMPOSSIBLE WITHOUT FULL-ON LOADING ALL SNAPSHOTS BACK IN.
    #
    #
    # # trying the same grind (but full) with old run/interation routine
    # mstar_min = np.min(mstar_and_SFR[:, 0])
    # mstar_max = np.max(mstar_and_SFR[:, 0])
    # SFR_min = np.min(mstar_and_SFR[:, 1])
    # SFR_max = np.max(mstar_and_SFR[:, 1])
    #
    # mstar_arr = np.linspace(mstar_min, mstar_max, 40, endpoint=True)
    # SFR_arr = np.linspace(SFR_min, SFR_max, 40, endpoint=True)
    #
    # print("mstar_arr", mstar_arr)
    # print("SFR_arr", SFR_arr)
    #
    # test_multiple_evolution_timedirection_andback_haloscaling_mgasfromSFR(
    #     zstart=0.,
    #     zend=z_at_value(cosmo.age, cosmo.age(0) - 0.8 * u.Gyr),
    #     timestep=1.e-3,
    #     silent=True,
    #     write_todisk=1,
    #     vary_params={
    #         'fgal': (0.1,),
    #         'SFE': (1.,),
    #         'MLF': (0.1,),
    #         'HLF': (0.1,),
    #         'mstar': 10**mstar_arr,
    #         'SFR': 10**(SFR_arr + 9.)
    #     },
    #     directions=(
    #         {'name': 'backward', 'factor': -1},
    #         # {'name': 'forward', 'factor': 1}
    #     )
    # )
    return


if __name__ == "__main__":
    start_time = timer()
    main()
    end_time = timer()
    delta_time_seconds = end_time - start_time
    print("Total run time:", timedelta(seconds=delta_time_seconds), "(hours:minutes:seconds)",
          f"\n({delta_time_seconds} seconds)")
else:
    # print('Importing main.py of SiGMo. '  # deactivated due to multiprocessing: main.py gets imported for evey thread?
    #       'This file should usually be executed. '
    #       'Are you sure you just want to import it?')
    pass
