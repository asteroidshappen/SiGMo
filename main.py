import SiGMo_classes as sgm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def testrun_1(timestep: float = 1.e-3, silent: bool = False):
    EnvA = sgm.Environment(name='Environment A')
    if not silent:
        print(f"Created '{EnvA.name}'")
    GalA1 = EnvA.create_Galaxy(name='Galaxy A1')
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
    for t in range(1000):
        EnvA.reference_evolve(timestep=timestep)
        GalA1_history.append(GalA1.make_snapshot())
        if not silent:
            print(f"{t+1}, z={EnvA.z}, lookbacktime={EnvA.lookbacktime}")

    if not silent:
        print(EnvA)
        print(GalA1)


    timesteps_l = np.linspace(0, len(GalA1_history) * timestep, len(GalA1_history))
    mgas_l = [GalA1_history[i]['mgas'] for i, item in enumerate(GalA1_history)]
    mstar_l = [GalA1_history[i]['mstar'] for i, item in enumerate(GalA1_history)]
    mout_l = [GalA1_history[i]['mout'] for i, item in enumerate(GalA1_history)]
    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(timesteps_l, mgas_l)
    # ax.plot(timesteps_l, mstar_l)
    # ax.plot(timesteps_l, mout_l)
    # ax.set_yscale('log')
    fig.show()

    return


def main():
    testrun_1()
    return


if __name__ == "__main__":
    main()
else:
    print('Importing main.py of SiGMo. '
          'This file should usually be executed. '
          'Are you sure you just want to import it?')