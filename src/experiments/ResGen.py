from copy import deepcopy
import requests
from os import getenv
import numpy as np

IONQ_API_KEY = getenv("IONQ_API_KEY")


def genKeys(n, dim):
    # n: discretization num
    # dim: dimension

    keys = []
    binString = ["0" for _ in range(n * dim)]

    if dim == 1:
        for i in range(n):
            bsNow = deepcopy(binString)
            n1 = i % n
            bsNow[n - n1 - 1] = "1"
            bs = "".join(x for x in bsNow)
            keys.append(bs)

    if dim == 2:
        for i in range(n**dim):
            bsNow = deepcopy(binString)
            n1 = i % n
            n2 = i // n
            bsNow[n - n2 - 1] = "1"
            bsNow[2 * n - n1 - 1] = "1"
            bs = "".join(x for x in bsNow)
            keys.append(bs)
    return keys


def genKeys_ob(n):
    keys = []
    binString = ["0" for _ in range(n + 1)]

    for i in range(n):
        bsNow = deepcopy(binString)
        bsNow[n - i] = "1"
        bs = "".join(x for x in bsNow)
        keys.append(bs)
    binString[0] = "1"
    for i in range(n):
        bsNow = deepcopy(binString)
        n1 = i % n
        bsNow[n - i] = "1"
        bs = "".join(x for x in bsNow)
        keys.append(bs)
    return keys


def genResDict(n, dim):
    res = {}
    keys = genKeys(n, dim)
    for key in keys:
        res[key] = []

    return res


def genResDict_ob(n):
    res = {}
    keys = genKeys_ob(n)
    for key in keys:
        res[key] = []

    return res


def resultGen(resList, n, dim):
    resDict = genResDict(n, dim)
    for res in resList:
        for key in resDict.keys():
            try:
                resDict[key].append(res[str(key)])
            except KeyError:
                resDict[key].append(0)
    return resDict


def resultDictGen(iqList, n, dim):
    resDict = genResDict(n, dim)
    for iq in iqList:
        res = iq.results()
        for key in resDict.keys():
            try:
                resDict[key].append(res[str(key)])
            except KeyError:
                resDict[key].append(0)
    return resDict


def resultDictGen_ob(iqList, n):
    resDict = genResDict_ob(n)
    for iq in iqList:
        res = iq.results()
        for key in resDict.keys():
            try:
                resDict[key].append(res[str(key)])
            except KeyError:
                resDict[key].append(0)
    return resDict


def chi(x, y):
    if x == y:
        return 1
    else:
        return 0


def sampleFromList(l):
    # assume list l only contains positive elements
    # sample some element c_j with probability c_j/norm_1(l)
    l = np.array(l)
    lnorm = sum(l)
    l = l / lnorm
    lcdf = np.zeros_like(np.array(l))
    lcdf[0] = l[0]
    for i in range(1, len(l)):
        lcdf[i] = lcdf[i - 1] + l[i]
    r = np.random.uniform(0, 1)
    return np.argmax(r < lcdf)


def resfromId(job_id, num):
    headers = {
        "Authorization": "apiKey " + IONQ_API_KEY,
    }
    response = requests.get("https://api.ionq.co/v0.2/jobs/" + job_id, headers=headers)
    res = response.json()
    if res["status"] != "completed":
        print("Job is not completed")
        print(res)
        return None
    else:
        resDict = {}
        print(res["gate_counts"])
        res = res["data"]["histogram"]
        for key in res.keys():
            newkey = str(bin(int(key)))[2:]
            l = len(newkey)
            zs = "0" * (num - l)
            newkey = zs + newkey
            resDict[newkey[::-1]] = res[key]
            # res[bin(int(key))] = res.pop(key)
        return resDict
