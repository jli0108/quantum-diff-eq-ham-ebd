from simuq import QSystem
from simuq import Qubit
from simuq.environment import Qubit
from simuq.environment import Qubit
from simuq.qutip import QuTiPProvider
from simuq.ionq import IonQProvider
import numpy as np
from os import getenv

IONQ_API_KEY = getenv("IONQ_API_KEY")


class advectionSolver:
    def __init__(
        self, dimension, constList=None, variableCoeffFunc=None, primeFunc=None
    ):
        # super().__init__()
        self.dimension = dimension
        self.constList = constList  # [c_1, c_2, \cdots, c_n]
        self.variableCoeffFunc = variableCoeffFunc
        self.primeFunc = primeFunc

    def solve(
        self,
        discretizeNum,
        timeLength,
        provider,
        trotterNum,
        trotterOrder,
        circ=None,
        onSimulator=True,
        noise=False,
    ):

        self.iq = IonQProvider(api_key=IONQ_API_KEY)
        self.qtp = QuTiPProvider()
        self.discretizeNum = discretizeNum  # same discretization number in each axis
        self.qubitNum = self.discretizeNum * self.dimension  # Here Nd qubits are enough
        self.spaceStep = 1 / self.discretizeNum
        self.timeLength = timeLength
        self.qs = QSystem()
        self.qubitList = [Qubit(self.qs) for _ in range(self.qubitNum)]
        self.provider = provider
        self.trotterNum = trotterNum
        self.trotterOrder = trotterOrder
        self.noise = noise
        self.nodes = np.array(range(self.discretizeNum + 1)) / self.discretizeNum
        self.midNodes = np.zeros(self.discretizeNum)
        for k in range(self.discretizeNum):
            self.midNodes[k] = (self.nodes[k] + self.nodes[k + 1]) / 2
        self.H = self.HamiltonianConstruct()
        if self.provider == "qutip":
            self.initState = circ
        elif self.provider == "ionq":
            self.circ = circ

        self.qs.add_evolution(self.H, timeLength)

        if provider == "ionq":
            if self.circ != None:
                self.iq.compile(
                    self.qs,
                    aais="2pauli",
                    state_prep=self.circ,
                    trotter_num=self.trotterNum,
                    trotter_mode=self.trotterOrder,
                    backend="aria-1",
                )
            else:
                print("-----", "Set initial state as 0", "-----")
                self.iq.compile(self.qs, aais="2pauli")
            self.iq.run(
                shots=300, on_simulator=onSimulator, verbose=-1, with_noise=self.noise
            )
            return self.iq

        if provider == "qutip":
            if self.initState != None:
                self.qtp.compile(self.qs, initial_state=self.initState)
            else:
                print("-----", "Set initial state as 0", "-----")
                self.qtp.compile(self.qs)
            self.qtp.run()
            # qutipResults = qtp.results()
            return self.qtp

        else:
            raise BaseException

    def HamiltonianConstruct(self):
        # variable coeff is a scalar function

        if self.variableCoeffFunc == None:
            if self.dimension == 1:
                H = 0
                for i in range(self.discretizeNum - 1, -1, -1):
                    H -= (self.constList[0] * self.discretizeNum / 4) * (
                        self.qubitList[i].Y * self.qubitList[i - 1].X
                        - self.qubitList[i].X * self.qubitList[i - 1].Y
                    )
            #    H = -(0.25 / self.spaceStep) * (self.qubitList[0].X * self.qubitList[self.qubitNum-1].Y - self.qubitList[0].Y * self.qubitList[self.qubitNum-1].X)
            #    for i in range(0,self.qubitNum-1):
            #        H = H + (0.25 / self.spaceStep) * (self.qubitList[i].X * self.qubitList[i+1].Y - self.qubitList[i].Y * self.qubitList[i+1].X)
            #    H = self.constList[0] * H

            elif self.dimension == 2:
                H = 0
                for i in range(self.discretizeNum - 1, -1, -1):
                    H -= (self.constList[0] * self.discretizeNum / 4) * (
                        self.qubitList[i].Y
                        * self.qubitList[(i - 1) % self.discretizeNum].X
                        - self.qubitList[i].X
                        * self.qubitList[(i - 1) % self.discretizeNum].Y
                    )
                    H -= (self.constList[1] * self.discretizeNum / 4) * (
                        self.qubitList[i + self.discretizeNum].Y
                        * self.qubitList[
                            self.discretizeNum + (i - 1) % self.discretizeNum
                        ].X
                        - self.qubitList[self.discretizeNum + i].X
                        * self.qubitList[
                            self.discretizeNum + (i - 1) % self.discretizeNum
                        ].Y
                    )
                # H = -(0.25 * self.constList[0] / self.spaceStep) * (self.qubitList[0].X * self.qubitList[self.discretizeNum-1].Y - self.qubitList[0].Y * self.qubitList[self.discretizeNum-1].X)
                # H = H - (0.25 * self.constList[1] / self.spaceStep) * (self.qubitList[self.discretizeNum].X * self.qubitList[self.qubitNum-1].Y - self.qubitList[self.discretizeNum].Y * self.qubitList[self.qubitNum-1].X)
                # for i in range(0,self.discretizeNum-1):
                #    j = i + self.discretizeNum
                #    H = H + (0.25 * self.constList[0] / self.spaceStep) * (self.qubitList[i].X * self.qubitList[i+1].Y - self.qubitList[i].Y * self.qubitList[i+1].X)
                #    H = H + (0.25 * self.constList[1] / self.spaceStep) * (self.qubitList[j].X * self.qubitList[j+1].Y - self.qubitList[j].Y * self.qubitList[j+1].X)
        else:
            if self.dimension == 1:
                H = 0
                for i in range(self.discretizeNum - 1, -1, -1):
                    inx = self.discretizeNum - 1 - i
                    H -= (
                        self.variableCoeffFunc(self.midNodes[inx])
                        * self.discretizeNum
                        / 4
                    ) * (
                        self.qubitList[i].Y * self.qubitList[i - 1].X
                        - self.qubitList[i].X * self.qubitList[i - 1].Y
                    )
                # H = -(0.25 * self.variableCoeffFunc(self.midNodes[-1]) / self.spaceStep) * (self.qubitList[0].X * self.qubitList[self.qubitNum-1].Y - self.qubitList[0].Y * self.qubitList[self.qubitNum-1].X)
                # for i in range(0,self.qubitNum-1):
                #    H = H + (0.25 * self.variableCoeffFunc(self.midNodes[i]) / self.spaceStep) * (self.qubitList[i].X * self.qubitList[i+1].Y - self.qubitList[i].Y * self.qubitList[i+1].X)

            elif self.dimension == 2:
                H = 0
                for i in range(self.discretizeNum - 1, -1, -1):
                    inx = self.discretizeNum - 1 - i
                    H -= (
                        self.variableCoeffFunc(self.midNodes[inx])
                        * self.discretizeNum
                        / 4
                    ) * (
                        self.qubitList[i].Y
                        * self.qubitList[(i - 1) % self.discretizeNum].X
                        - self.qubitList[i].X
                        * self.qubitList[(i - 1) % self.discretizeNum].Y
                    )
                    H -= (
                        self.variableCoeffFunc(self.midNodes[inx])
                        * self.discretizeNum
                        / 4
                    ) * (
                        self.qubitList[i + self.discretizeNum].Y
                        * self.qubitList[
                            self.discretizeNum + (i - 1) % self.discretizeNum
                        ].X
                        - self.qubitList[self.discretizeNum + i].X
                        * self.qubitList[
                            self.discretizeNum + (i - 1) % self.discretizeNum
                        ].Y
                    )
                # H =     -(0.25 * self.variableCoeffFunc(self.midNodes[-1]) / self.spaceStep) * (self.qubitList[0].X * self.qubitList[self.discretizeNum-1].Y - self.qubitList[0].Y * self.qubitList[self.discretizeNum-1].X)
                # H = H   -(0.25 * self.variableCoeffFunc(self.midNodes[-1]) / self.spaceStep) * (self.qubitList[self.discretizeNum].X * self.qubitList[self.qubitNum-1].Y - self.qubitList[self.discretizeNum].Y * self.qubitList[self.qubitNum-1].X)
                # for i in range(0,self.discretizeNum-1):
                #    j = i + self.discretizeNum
                #    H = H + (0.25 * self.variableCoeffFunc(self.midNodes[i])  / self.spaceStep) * (self.qubitList[i].X * self.qubitList[i+1].Y - self.qubitList[i].Y * self.qubitList[i+1].X)
                #    H = H + (0.25 * self.variableCoeffFunc(self.midNodes[i])  / self.spaceStep) * (self.qubitList[j].X * self.qubitList[j+1].Y - self.qubitList[j].Y * self.qubitList[j+1].X)

        return H
