import SiGMo_classes as sgm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def main():
    # testrun_1(silent=True)
    # testrun_2(silent=True)
    testrun_3(silent=True)
    return


if __name__ == "__main__":
    main()
else:
    print('Importing main.py of SiGMo. '
          'This file should usually be executed. '
          'Are you sure you just want to import it?')