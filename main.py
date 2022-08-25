import SiGMo as sgm

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from timeit import default_timer as timer
from datetime import timedelta, datetime

# ====================
# set some global vars

alphabet_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# =====================
# some helper functions

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
    ax_obs.plot(GMS_x, sgm.GMS_Saintonge2016(GMS_x),
                color='xkcd:purplish pink', ls=':', label="Saintonge et al. 2016")
    ax_obs.plot(GMS_x, sgm.GMS_Saintonge2022(GMS_x),
                color='xkcd:magenta', ls='--', label="Saintonge & Catinella 2022")
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
    ax_obs.legend(title=" Galaxy Main Sequence at 0.01 < z < 0.05:  ", loc='lower right')
    fig.savefig(plot_dir / f'mstar_SFR_obs_vs_ICs_of_sims_{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}.png')
    return fig, ax_obs

def explore_attrs_and_augment_plot(env, mstar_mesh, sfr_mesh, fig, ax_obs, plot_dir):
    """
    Helper routine that allows to visually explore the attributes/quantities of a Galaxy-Halo pair based on user input.
    :param env: The AstroObjects.Environment instance that in itself holds references to the Halo and Galaxy objects
    :param mstar_mesh: A mesh of stellar mass values; designed to take the mstar bin mesh from the binned SDSS data
    :param sfr_mesh: A mes of star-formation rate values; designed to take the SFR bin mesh from the binned SDSS data
    :param fig: The figure of the already created plot that is to be augmented with additional info based on user input
    :param ax_obs: The main axis of the already created plot; a nex axis will be inserted to house the additional data
    :param plot_dir: The directory on disc whereto save the finished plot
    :return: fig, ax_obs, ax_values: figure, main existing axes, new additional-info axes

    Key info: This routine, as it is currently being used, only properly works for SDSS, since there the mstar_mesh and
    sfr_mesh match the created AstroObjects, whose data can be displayed. In the case of xCG, for example, this is not
    true, and the plotting will stop after drawing the lines to the (not truly data matching, but only mesh matching)
    bin and trying (via Exception handling) to check whether the data exists; the printed message reads:
    'The mstar-SFR bin you selected does not contain a Galaxy/Halo. Please restart and try again'
    """
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
        return fig, ax_obs, ax_values


# ================
# various testruns
# (they were removed as the routine evolved a lot and they are now obsolete)



def main():
    # setting paths
    project_dir = Path.cwd()
    sfr79_dir = project_dir / 'data' / "SFR79_grids"
    date_and_time_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    plot_dir = project_dir / 'plots' / '_tmp' / date_and_time_str
    plot_dir.mkdir()
    out_dir = project_dir / 'outputs' / '_tmp' / date_and_time_str  # only defining the path, not creating yet


    # SDSS
    # reading observational results
    sfr79_medians = np.loadtxt(str(sfr79_dir / "SFR79_2dhist_medians.txt"))
    mstar_mesh =  np.loadtxt(str(sfr79_dir / "SFR79_2dhist_binedges_mstar_mesh.txt"))
    sfr_mesh = np.loadtxt(str(sfr79_dir / "SFR79_2dhist_binedges_sfr_mesh.txt"))
    n_binned = np.loadtxt(str(sfr79_dir / "SFR79_2dhist_binnumbers.txt"))

    # remove bins with less than n objects in them
    n_binned_min = 40
    sfr79_medians = np.where(n_binned >= n_binned_min, sfr79_medians, np.nan)

    # getting the mstar-SFR bin cells that do have observational results
    mstar_and_SFR = sgm.calc_bincentres_where_not_nan(sfr79_medians, mstar_mesh, sfr_mesh)


    # xCOLD GASS
    # reading the xCG + SDSS + additional derived quantities catalogue
    xCG_df = pd.read_csv(sfr79_dir / "xCOLD_GASS_with_SDSS_SFR79_df.csv")
    xCG_minimal_selector = ((xCG_df.LOGMSTAR > -900) & (xCG_df.LOGSFR_BEST > -900) & (xCG_df.LOGMH2 > - 900))
    xCG_mstar = np.squeeze(xCG_df.loc[xCG_minimal_selector & (xCG_df['FLAG_CO'] == 1), 'LOGMSTAR'].to_numpy())
    xCG_SFR = np.squeeze(xCG_df.loc[xCG_minimal_selector & (xCG_df['FLAG_CO'] == 1), 'LOGSFR_BEST'].to_numpy())
    xCG_mgas = np.squeeze(xCG_df.loc[xCG_minimal_selector & (xCG_df['FLAG_CO'] == 1), 'LOGMH2'].to_numpy())
    # # If you want to use gas limits same as proper gas measurements, use this instead:
    # # instead of straight mgas use derived gas mass fraction, because it uses full  measurement or limit, accordingly
    # xCG_mgas = np.squeeze(xCG_df['gas_mass_fraction'].to_numpy() * xCG_df['LOGMSTAR'].to_numpy())  # conv frm fraction


    # use data from SDSS or xCG as initial conditions?
    use_as_ICs = input(f"Specify which data to base the galaxies on: (SDSS/xCG/GMS/GMS_GAUSS)")

    # intial values for Galaxy properties
    if use_as_ICs.casefold() == "SDSS".casefold():  # SDSS
        mstar = 10**mstar_and_SFR[:, 0]
        SFR = 10**mstar_and_SFR[:, 1] * 10**9   # CONVERSION of 'per yr' (obs) to 'per Gyr' (sims)
        sSFR = SFR / mstar
        mgas = sgm.calculate_mgas_mstar_from_sSFR_Saintonge2022(sSFR / 10 ** 9, log_values=False, withscatter=False) * mstar
        SFE = SFR / mgas   # set SFE to one (a.t.m. const) unique value, in harmony with the sSFR relation (through mgas)
        z = 0.
    elif use_as_ICs.casefold() == "xCG".casefold():  # xCG
        mstar = 10**xCG_mstar
        SFR = 10**xCG_SFR * 10**9   # CONVERSION of 'per yr' (obs) to 'per Gyr' (sims)
        sSFR = SFR / mstar
        mgas = 10**xCG_mgas
        SFE = SFR / mgas   # set SFE to one (a.t.m. const) unique value, in harmony with the sSFR relation (through mgas)
        z = 0.
    elif use_as_ICs.casefold() == "GMS".casefold():  # galaxies on GMS or off GMS, at redshift to be specified
        # input
        try:
            mstar_min = float(input(f"Lowest stellar mass of the galaxies: (log(mstar/M☉), default: 7)") or 7.)
            mstar_max = float(input(f"Highest stellar mass of the galaxies: (log(mstar/M☉), default: 12)") or 12.)
            mstar_n = int(input(f"Number of galaxies between "
                                f"log(mstar/M☉) = {mstar_min} to {mstar_max}: (default: 50)") or 50)
            z = float(input(f"Redshift of the GMS: (default: 0)") or 0.)
            SFR_offset = float(input(f"Offset ΔSFR of galaxies from the GMS: (log(ΔSFR/M☉ yr⁻¹), default: 0)") or 0.)
        except ValueError:
            print("Some input value(s) could not be converted to numeric value(s)")
            mstar_min, mstar_max, mstar_n, z, SFR_offset = tuple([None] * 5)  # will crash the code in the next lines

        # calc IC values
        mstar_log = np.linspace(mstar_min, mstar_max, mstar_n)
        sfr_log = sgm.GMS_Leslie2020(mstar_log, z=z, log=True)
        mstar = 10**mstar_log
        SFR = 10**(sfr_log + SFR_offset) * 10**9   # CONVERSION of 'per yr' (obs) to 'per Gyr' (sims)
        sSFR = SFR / mstar
        mgas = sgm.calculate_mgas_mstar_from_sSFR_Tacconi2020(sSFR=sSFR,
                                                              mstar=mstar,
                                                              z=z,
                                                              log=False,
                                                              withscatter=False) * mstar
        SFE = SFR / mgas
        # SFE = np.array([1.5] * len(mstar))
        # mgas = SFR / SFE
    elif use_as_ICs.casefold() == "GMS_GAUSS".casefold():
        # input
        try:
            mstar_min = float(input(f"Lowest stellar mass of the galaxies: (log(mstar/M☉), default: 6)") or 6.)
            mstar_max = float(input(f"Highest stellar mass of the galaxies: (log(mstar/M☉), default: 10)") or 10.)
            n_gal = int(input(f"Number of galaxies between "
                                f"log(mstar/M☉) = {mstar_min} to {mstar_max}: (default: 100)") or 100)
            z = float(input(f"Redshift of the GMS: (default: 2)") or 2.)
            SFR_offset = float(input(f"Offset ΔSFR of galaxies from the GMS: (log(ΔSFR/M☉ yr⁻¹), default: 0)") or 0.)
            sfr_sigma = float(input(f"Standard deviation of the Gaussian SFR distribution: (dex, default: 0.1)") or 0.1)
        except ValueError:
            print("Some input value(s) could not be converted to numeric value(s)")
            mstar_min, mstar_max, n_gal, z, SFR_offset, sfr_sigma = tuple([None] * 6)  # will crash the code in the next lines

        # calc IC values

        # initiate rng
        rng = np.random.default_rng(12345)

        # two random dist
        mstar_log = rng.uniform(low=mstar_min,
                                high=mstar_max,
                                size=n_gal)
        sfr_normal_log = rng.normal(loc=SFR_offset,
                             scale=sfr_sigma,
                             size=n_gal)

        # calc GMS and add it to the SFR normal dist
        gms_sfr_log = sgm.GMS_Leslie2020(mstar_log, z=z, log=True)
        sfr_log = sfr_normal_log + gms_sfr_log

        # convert from log and calc further data
        mstar = 10**mstar_log
        SFR = 10**sfr_log * 10**9   # CONVERSION of 'per yr' (obs) to 'per Gyr' (sims)
        sSFR = SFR / mstar
        mgas = sgm.calculate_mgas_mstar_from_sSFR_Tacconi2020(sSFR=sSFR,
                                                              mstar=mstar,
                                                              z=z,
                                                              log=False,
                                                              withscatter=False) * mstar
        SFE = SFR / mgas

        # control plot?
        plotting = True
        if plotting:
            fig_gauss, ax_gauss = plt.subplots(1, 2, figsize=(12, 5))

            ax_gauss[0].axhline(SFR_offset, ls='--', color='xkcd:green', zorder=-1, label=f'In: mean={SFR_offset:.3f}, std={sfr_sigma:.3f}')
            ax_gauss[0].scatter(mstar_log, sfr_normal_log, label=f'Out: mean={np.mean(sfr_normal_log):.3f}, std={np.std(sfr_normal_log):.3f}, N={n_gal}')

            _mstar_lin = np.linspace(mstar_min, mstar_max, 1000)
            ax_gauss[1].plot(_mstar_lin, sgm.GMS_Leslie2020(mstar=_mstar_lin, z=z, log=True), ls='--', color='xkcd:green', zorder=-1, label=f'Leslie+20 GMS at z={z}')
            ax_gauss[1].scatter(mstar_log, sfr_log, label=f'Transformed to GMS')

            for _ax in ax_gauss:
                _ax.set_xlabel(r'log $M_\mathrm{star}$ [$M_\odot$]')
                _ax.set_ylabel(r'log $SFR$ [$M_\odot$ yr$^{-1}$]')
                _ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
                _ax.legend()

            fig_gauss.savefig(plot_dir / f'Generate_Distribution_around_GMS_{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}.png', dpi=300)


    elif isinstance(use_as_ICs, str):
        raise ValueError
    else:
        raise TypeError


    # change the overall accretion scaling (aka sMIR_scaling) from the default 1?
    sMIR_scaling_basefactor = float(input(f"Enter off-GMS sMIR accretion scaling: (Gyr⁻¹ / dex ΔSFR, default: 1)") or 1.)
    if sMIR_scaling_basefactor == 1:
        sMIR_scaling = np.array([1.] * len(mstar))
    else:
        sfr_gms = sgm.GMS_Leslie2020(mstar, z=z, log=False) * 10**9   # conversion from yr⁻¹ to Gyr⁻¹ (like SFR)
        delta_sfr_log = np.log10(SFR / sfr_gms)
        sMIR_scaling = np.array([1.] * len(mstar)) * (sMIR_scaling_basefactor**delta_sfr_log)


    # set an sMIR_scaling_updater (if desired; if not, set to None)
    sMIR_scaling_updater = np.array([sgm.sMIR_scaling_updater_deltaGMS] * len(mstar))


    # SFE = np.array([1.] * len(mstar))
    # SFE = SFR / mgas   # set SFE to one (a.t.m. const) unique value, in harmony with the sSFR relation (through mgas)
    fgal = np.array([0.4] * len(mstar))  # following Lilly+13
    # fgal = np.array([0.3] * len(mstar))  # slightly lower accretion than Lilly+13
    # fgal = np.array([0.5] * len(mstar))  # slightly higher accretion than Lilly+13
    # fgal = np.array([0.1] * len(mstar))  # significantly lower than Lilly+13
    # fgal = np.array([0.01] * len(mstar))  # ridiculously much lower than Lilly+13
    MLF = np.array([0.1] * len(mstar))
    # MLF = np.array([0.05] * len(mstar))
    # MLF = np.array([0.2] * len(mstar))

    # initial values for Halo properties
    mhalo = np.array([sgm.iter_mhalo_from_mstar(mstar_i, z=z, try_lookup=False, interpolate=True) for mstar_i in mstar])
    # BDR = np.array([0.2] * len(mstar))
    BDR = np.array([0.15] * len(mstar))  # actual value from Lilly+13
    HLF = np.array([0.1] * len(mstar))

    # # initial values of Environment properties
    # z = 0. if z is None else z  # set z = 0. if no other value has been set yet (e.g. through input)


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
    IC_halo_sMIR_scaling = sgm.IC.single_param('sMIR_scaling', sMIR_scaling)
    IC_halo_sMIR_scaling_basefactor = sgm.IC.single_param('sMIR_scaling_basefactor', np.array([sMIR_scaling_basefactor] * len(mstar)))
    IC_halo_sMIR_scaling_updater = sgm.IC.single_param('sMIR_scaling_updater', sMIR_scaling_updater)

    # Environment IC
    IC_env_zstart = sgm.IC.single_param('zstart', [z])  # can also use 'lookbacktime' (Gyrs) instead of zstart
    # IC_env_lookbacktime = sgm.IC.single_param('lookbacktime', [0.8])  # THIS IS FOR FORWARDS, THEN BACKWARDS COMPARISON

    # combine into one IC object for Environment, one for Halos and one for Galaxies
    IC_env_comb = (IC_env_zstart)
    # IC_env_comb = (IC_env_lookbacktime)  # THIS IS FOR FORWARDS, THEN BACKWARDS COMPARISON
    IC_halo_comb = (IC_halo_mtot +
                    IC_halo_BDR +
                    IC_halo_mdm +
                    IC_halo_mgas +
                    IC_halo_HLF +
                    IC_halo_sMIR_scaling +
                    IC_halo_sMIR_scaling_basefactor +
                    IC_halo_sMIR_scaling_updater)
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


        # # THIS IS JUST BACKWARDS INTEGRATION FROM z=0, BUT FARTHER
        # 
        # # create BACKWARD integrator
        # Integrator = sgm.FTI(
        #     env=env,
        #     evolve_method='evolve',
        #     dt=-1.e-3,
        #     # dt=-1.e-4,
        #     t_start=env.lookbacktime,
        #     t_end=2
        # )
        # 
        # # run the BACKWARD integrator
        # print("Starting integration")
        # Integrator.integrate(
        #     wtd=1,
        #     # wtd=10,
        #     outdir=out_dir / "0_backward_2Gyr_dt1e-3",
        #     # outdir=out_dir / "0_backward_2Gyr_dt1e-4_wtd10",
        #     single_snapshots=False
        # )



        # # THIS IS JUST FORWARDS INTEGRATION FROM z SET EARLIER, RUNNING FOR 1 Gyr
        #
        # # create BACKWARD integrator
        # Integrator = sgm.FTI(
        #     env=env,
        #     evolve_method='evolve',
        #     dt=1.e-3,
        #     t_start=env.lookbacktime,
        #     t_end=env.lookbacktime - 1.
        # )
        #
        # # run the BACKWARD integrator
        # print("Starting integration")
        # Integrator.integrate(
        #     wtd=1,
        #     # wtd=10,
        #     outdir=out_dir / f"0_forward_from_z{z}_1Gyr_dt1e-3_SFRoffset{SFR_offset}",
        #     single_snapshots=False
        # )



        # # THIS IS FORWARDS INTEGRATION FROM z SET EARLIER, RUNNING UNTIL z=0
        # 
        # # create BACKWARD integrator
        # Integrator = sgm.FTI(
        #     env=env,
        #     evolve_method='evolve',
        #     dt=1.e-3,
        #     t_start=env.lookbacktime,
        #     t_end=0.
        # )
        # 
        # # run the BACKWARD integrator
        # print("Starting integration")
        # Integrator.integrate(
        #     wtd=1,
        #     # wtd=10,
        #     outdir=out_dir / f"0_forward_from_z{z}_to_z0_dt1e-3_SFRoffset{SFR_offset}",
        #     single_snapshots=False
        # )


        # # THIS (again) IS JUST BACKWARDS INTEGRATION FROM z=0, BUT FARTHER and with ΔSFR dependent sMIR scaling
        #
        # # create BACKWARD integrator
        # Integrator = sgm.FTI(
        #     env=env,
        #     evolve_method='evolve',
        #     dt=-1.e-3,
        #     # dt=-1.e-4,
        #     t_start=env.lookbacktime,
        #     t_end=2
        # )
        #
        # # run the BACKWARD integrator
        # print("Starting integration")
        # Integrator.integrate(
        #     wtd=1,
        #     # wtd=10,
        #     outdir=out_dir / f"0_backward_2Gyr_dt1e-3_sMIR_scaling_basefactor{sMIR_scaling_basefactor}",
        #     # outdir=out_dir / "0_backward_2Gyr_dt1e-4_wtd10",
        #     single_snapshots=False
        # )



        # THIS (again) IS FORWARDS INTEGRATION FROM z SET EARLIER, RUNNING UNTIL z=0

        # create BACKWARD integrator
        Integrator = sgm.FTI(
            env=env,
            evolve_method='evolve',
            dt=1.e-3,
            t_start=env.lookbacktime,
            t_end=0.
        )

        # run the BACKWARD integrator
        print("Starting integration")
        Integrator.integrate(
            wtd=1,
            outdir=out_dir / f"0_forward_from_z{z}_to_z0_dt1e-3_SFRoffset{SFR_offset}_sMIR_scaling_basefactor{sMIR_scaling_basefactor}",
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
