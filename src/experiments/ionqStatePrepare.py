import numpy as np
from simuq.ionq.ionq_api_circuit import IonQAPICircuit


class ionqSP:
    def __init__(self, discretizationNum, dimension, control=None):

        self.discretizationNum = discretizationNum
        self.dimension = dimension
        self.sysSize = discretizationNum * dimension

        if control == None:
            self.circuit = IonQAPICircuit(self.sysSize)
        else:
            self.circuit = IonQAPICircuit(self.sysSize + 1)

    def statePrepAux(self, startPoint, aList, reverse=False):
        # Generic Solution
        # assume startPoint = 0
        # aList is a numpy array
        aList = aList / np.linalg.norm(aList)

        if len(aList) > 1:
            half = (len(aList) - 1) // 2
            aList_L = aList[0 : half + (len(aList) + 1) % 2]
            aList_R = aList[half + (len(aList) + 1) % 2 : len(aList)]
            if reverse == False:
                # starts with 000...01
                # if np.linalg.norm(aList_L) > 1e-5:
                self.partialSWAP(
                    startPoint + half,
                    startPoint + len(aList) - 1,
                    np.arccos(np.linalg.norm(aList_L)),
                )
                if len(aList_L) > 1 and np.linalg.norm(aList_L) > 1e-5:
                    self.statePrepAux(
                        startPoint + half + 1, aList_L / np.linalg.norm(aList_L)
                    )
                if len(aList_R) > 1 and np.linalg.norm(aList_R) > 1e-5:
                    self.statePrepAux(startPoint, aList_R / np.linalg.norm(aList_R))
            else:
                print("wrong.")
                pass
                # starts with 100...00
                # if sum(abs(aList_R)) > 1e-5:
                #    self.partialSWAP(startPoint, startPoint+half,-2*np.arccos(np.linalg.norm(aList_R)))
                # if sum(abs(aList_R)) > 1e-5:
                #    self.statePrepAux(startPoint, aList_R/np.linalg.norm(aList_R),True)
                # if sum(abs(aList_L)) > 1e-5:
                #    self.statePrepAux(startPoint+half+1, aList_L/np.linalg.norm(aList_L),True)

    def partialSWAP(self, q0, q1, theta):
        self.circuit = self.circuit.ms(q0, q1, np.pi / 2, 0, theta).ms(
            q0, q1, 0, -np.pi / 2, theta
        )

    def GHZ(self, qubitInxList):
        self.circuit = self.circuit.hadamard(qubitInxList[0])
        for i in range(len(qubitInxList) - 1):
            self.circuit = self.circuit.cnot(qubitInxList[i], qubitInxList[i + 1])

    def split2D(self, aList, reverse=False):
        if reverse:
            self.circuit = self.circuit.gpi(0, 0).gpi(self.discretizationNum, 0)
            self.statePrepAux(0, aList, True)
        else:
            self.circuit = self.circuit.gpi(self.discretizationNum - 1, 0).gpi(
                self.sysSize - 1, 0
            )
            self.statePrepAux(0, aList)

    def init1D(self, aList):
        assert self.sysSize == self.discretizationNum
        self.circuit = self.circuit.gpi(self.sysSize - 1, 0)
        self.statePrepAux(0, aList)

    def GaussDistribution2D(self):
        self.GHZ(
            [
                self.discretizationNum // 2 - 1,
                self.discretizationNum - 1,
                self.discretizationNum + self.discretizationNum // 2 - 1,
                self.sysSize - 1,
            ]
        )
        self.circuit = self.circuit.gpi(self.discretizationNum // 2 - 1, 0).gpi(
            self.discretizationNum + self.discretizationNum // 2 - 1, 0
        )

        a1 = gaussAmp(self.discretizationNum, 0.25, 0.08)[
            0 : self.discretizationNum // 2 + (self.discretizationNum) % 2
        ]
        a2 = gaussAmp(self.discretizationNum, 0.75, 0.08)[
            self.discretizationNum // 2
            + (self.discretizationNum) % 2 : self.discretizationNum
        ]
        a1 = a1 / np.linalg.norm(a1)
        a2 = a2 / np.linalg.norm(a2)

        self.statePrepAux(0, a2)
        self.statePrepAux(self.discretizationNum // 2, a1)
        self.statePrepAux(self.discretizationNum, a2)
        self.statePrepAux(self.discretizationNum + self.discretizationNum // 2, a1)


def gaussAmp(n, m, s, double=False):
    # n: the discretization num along dimension
    # m: mean
    # s: sigma
    # assume n is 2^a for now
    # solve this problem in [0,1] x [0,1]
    # start at 1/n - 1 (Discretized Points)

    if double:
        amp = [np.exp(-(((i / n) - m) ** 2) / (2 * s * s)) for i in range(0, n)] * 2
        amp = np.array(amp)
        amp = amp / np.linalg.norm(amp)
    else:
        amp = np.array(
            [np.exp(-(((i / n) - m) ** 2) / (2 * s * s)) for i in range(0, n)]
        )
        amp = amp / np.linalg.norm(amp)

    return amp
