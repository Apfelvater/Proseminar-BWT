#####################################################
#                                                   #
#   An implementation of the approach on            #
#   bijective BWT (and ST) by Manfred Kufleitner.   #
#                                                   #
#   Author: Nicholas Leerman                        #
#                                                   #
#####################################################

import sys
from bwt_tools_lib import *

def cycles_to_lyndon(cycles, u) -> list[str]:
    '''Applies lambda_L on the cycles (C_1, ..., C_n) where C_1 starts with the smallest number.\n
    Returns Lyndon-Factorization (v_n, ..., v_1)'''
    factorization = []
    for cycle in cycles[::-1]:
        v = ""
        for i in l(cycle):
            v += u[i]
        factorization += v
    return factorization

def context_graph(u, pi, k = -1):

    def edges_of_context_graph():
        pass

    def vertices_of_context_graph():
        c = []
        for j in range(len(pi)):
            context = ""
            next_pos = j
            for i in range(k):
                next_pos = pi[next_pos]
                context += u[next_pos]
            c += [context]
        return set(c)

    V = vertices_of_context_graph()
    print(list(V))
    E = edges_of_context_graph()

    return (E, V)

def M(w, k = -1):
     conj_classes = conjugacy_classes(w)
     k_sort(conj_classes, k)
     return conj_classes

def LM(w, k = -1):
    lf = lyndon_factorization(w)
    v = reversed(lf)
    conj_classes = conjugacy_classes(*v)
    k_sort(conj_classes, k)
    return conj_classes

def BWT(w):
    res = ""
    for u in M(w):
        res += u[-1]
    return res

def BWTS(w):
    res = ""
    for u in LM(w):
        res += u[-1]
    return res

def ST(w, k):
    res = ""
    for u in M(w, k):
        res += u[-1]
    return res

def LST(w, k):
    res = ""
    for u in LM(w, k):
        res += u[-1]
    return res

def decode_BWT(L, i):
    F = L.sort()
    res = "" 
    raise NotImplementedError

def decode_BWTS(u, return_lyndon_factorization = False):
    pi = standard_permutation(u)
    cycles = find_cycles(pi)
    lf = cycles_to_lyndon(cycles, u)
    if return_lyndon_factorization:
        return lf
    else:
        return "".join(lf)

def decode_ST(k, L, i):
    raise NotImplementedError

def decode_LST(k, L):
    pi = standard_permutation(L)
    cycles = find_cycles(pi)
    graph = context_graph(L, pi, k)
    # Graph to w (??)

def test():
    test_sets = [{
        "w"       : "bcbccbcbcabbaaba",
        "k"       : 2,
        "bwt"     : "bacbbaaccacbbcbb",
        "bwts"    : "abababaccccbbcbb",
        "st"      : "bbacabaacccbbcbb",
        "lst"     : "abababaccccbbcbb"
    }]
    for test_set in test_sets:
        assert(BWT(test_set["w"]) == test_set["bwt"])

        assert(BWTS(test_set["w"]) == test_set["bwts"])
        assert(decode_BWTS(test_set["bwts"]) == test_set["w"])

        assert(ST(test_set["w"], test_set["k"]) == test_set["st"])

        assert(LST(test_set["w"], test_set["k"]) == test_set["lst"])
        assert(decode_LST(test_set["k"], test_set["lst"]) == test_set["w"])

    print("Test done!")

    sys.exit()

if __name__ == "__main__":
    # ---------- TEST ----------
    test()
    # -------- TEST END --------

    w1 = "GTTCTGGAGCCTAACG"
    w2 = "bcbccbcbcabbaaba"

    w = w2

    st_order = 3 

    print(f"Starting computation of bijective BWT (BWTS) and bijective ST of order {st_order} (LST) of word {w}...")
    bwt = BWT(w)
    print(f"BWT\t= {bwt}")

    bwts = BWTS(w)
    print(f"BWTS\t= {bwts}")

    st = ST(w, st_order)
    print(f"ST\t= {st}")

    lst = LST(w, st_order)
    print(f"LST\t= {lst}")
