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


def main():
    testrun_1(silent=True)
    return


if __name__ == "__main__":
    main()
else:
    print('Importing main.py of SiGMo. '
          'This file should usually be executed. '
          'Are you sure you just want to import it?')