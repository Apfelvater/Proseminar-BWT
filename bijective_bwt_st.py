#####################################################
#                                                   #
#   An implementation of the approach on            #
#   bijective BWT (and ST) by Manfred Kufleitner.   #
#                                                   #
#   Author: Nicholas Leerman                        #
#                                                   #
#####################################################

from bwt_tools_lib import *

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# <Misc>
def draw_graph(V, E):
    # G = (V, E)
    nx_G = nx.Graph()
    nx_G.add_nodes_from(V)
    for edge in E:
        nx_G.add_edge(edge[0], edge[2], weight=edge[1]+1) # edge = (c1, i, c2)
    pos = nx.spring_layout(nx_G, seed=7)
    nx.draw_networkx(nx_G, pos, arrows=False, with_labels=True)
    edge_labels = nx.get_edge_attributes(nx_G, "weight")
    nx.draw_networkx_edge_labels(nx_G, pos, edge_labels)
    plt.show()
# </Misc>

def decode_graph(G, start = 0):
    V, E, adj = G[0][:], G[1][:], G[2][:]

    curr_edge = E[start]
    E.remove(curr_edge)

    # Positions in L of symbols of reversed(w)
    positions = [curr_edge[1]]

    while(E):
        pass

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
    c = None # Contexts of u, pi
    
    def edges_of_context_graph():
        E = []          # e.g. [("aa", 1, "ba"), ...]
        adjacency = {context:[] for context in c}  # e.g. {"aa" : [("ba", 1), ...], ...}
        # Creating the edges
        for i in range(len(u)):
            edge = (c[i], i, u[i] + c[i][0])
            adjacency[edge[0]].append((edge[2], edge[1]))
            E.append(edge)
        return E, adjacency

    def contexts():
        c = []
        for j in range(len(pi)):
            context = ""
            next_pos = j
            for i in range(k):
                next_pos = pi[next_pos]
                context += u[next_pos]
            c += [context]
        return c

    c = contexts()  # These are including doubles! len(c) == len(u)
    E, adj = edges_of_context_graph()

    return (sorted(set(c)), E, adj)

def M(w, k = -1) -> list:
     conj_classes = conjugacy_classes(w)
     k_sort(conj_classes, k)
     return conj_classes

def LM(w, k = -1):
    lf = lyndon_factorization(w)
    v = reversed(lf)
    conj_classes = conjugacy_classes(*v)
    k_sort(conj_classes, k)
    return conj_classes

def Transform(w, k = -1):
    res = ""
    start_i = -1
    i = 0
    for u in M(w, k):
        if u == w:
            start_i = i
        res += u[-1]
        i += 1
    return res, start_i

BWT = lambda w: Transform(w)
ST = lambda w, k: Transform(w, k)

def BWTS(w):
    res = ""
    for u in LM(w):
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

def decode_LST(k, L, draw_context_graph = False):
    pi = standard_permutation(L)
    cycles = find_cycles(pi)
    graph = context_graph(L, pi, k)
    if draw_context_graph:
        draw_graph(graph[0], graph[1])
    #TODO: Graph to w


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
        assert(BWT(test_set["w"])[0] == test_set["bwt"])

        assert(BWTS(test_set["w"]) == test_set["bwts"])
        assert(decode_BWTS(test_set["bwts"]) == test_set["w"])

        assert(ST(test_set["w"], test_set["k"])[0] == test_set["st"])

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
