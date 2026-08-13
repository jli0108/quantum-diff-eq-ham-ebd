from advectionSolver import advectionSolver
from ionqStatePrepare import ionqSP
import numpy as np
from os import getenv

IONQ_API_KEY = getenv("IONQ_API_KEY")

if __name__ == "__main__":
    dim = 2
    size = 10
    tList = np.linspace(0, 0.2, 3)
    print(tList)
    sp = ionqSP(size, dim)
    sp.GaussDistribution2D()

    # Ideal simulation
    ad = advectionSolver(dim, [1, 1])
    idList = []
    for t in tList:
        print("time: %f, trotter number: %d" % (t, min(t // 0.05, 1)))
        iq = ad.solve(
            size,
            t,
            "ionq",
            trotterNum=min(int(t // 0.05), 1),
            trotterOrder=2,
            circ=sp.circuit,
            onSimulator=True,
            noise=False,
        )
        idList.append(iq.task["id"])
    with open("simulation.txt", "w") as f:
        for id in idList:
            f.write(f"{id}\n")

    # Real machine simulation
    adReal = advectionSolver(dim, [1, 1])
    idRealList = []
    for t in tList:
        print("time: %f, trotter number: %d" % (t, min(t // 0.05, 1)))
        iq = adReal.solve(
            size,
            t,
            "ionq",
            trotterNum=min(int(t // 0.05), 1),
            trotterOrder=2,
            circ=sp.circuit,
            onSimulator=False,
        )
        idRealList.append(iq.task["id"])
    with open("real.txt", "w") as f:
        for id in idRealList:
            f.write(f"{id}\n")
