# Tools for bijective BWT and ST

def right_shift(w):
    return w[-1] + w[:-1]
r = lambda w: w[-1:] + w[:-1]   # right shift of a str/list/...
l = lambda w: w[1:] + w[0:1]    # left shift of str/list/...

def conjugacy_class(w):
    v = w[:]
    m = [v]
    for i in range(len(v)-1):
        v = r(v)
        m.append(v)
    return m

def k_sort(m, k = -1):
    '''Sorts m by the first k chars'''
    if k < 0 or k > len(m):
        k = len(m)
    m.sort(key = (lambda w: w[:k]))
    return m

def conjugacy_classes(*v):
    m = []
    for w in v:
        m += conjugacy_class(w)
    return m

def lyndon_factorization(w):
    '''Computes the lyndon factorization of w using Duval's algorithm.\n
    Source: https://cp-algorithms.com/string/lyndon_factorization.html'''
    factorization = []
    n = len(w)
    i = 0
    while(i < n):
        j = i + 1
        k = i
        while(j < n and w[k] <= w[j]):
            if w[k] < w[j]:
                k = i
            else:
                k+=1
            j += 1
        # Append all the lyndon words found in previous steps to result:
        while (i <= k):
            # Lyndon word starts at i and has length j-k
            factorization.append(w[i:i+(j-k)])
            # Next Lyndon word (if more were found) starts at:
            i += j-k

    return factorization

def standard_permutation(u : str) -> list[int]:
    v = list(u)
    v.sort()
    pi = []
    j = 0
    # Outer loop: iterating through all letters sorted
    while (j < len(v)):
        # Inner loop: iterating through all letters of word u
        for i in range(len(u)):
            if u[i] == v[j]:
                pi += [i]
                j += 1
                if j >= len(v):
                    return pi
                # Symbol is changing
                if v[j] != v[j-1]:
                    break
    return pi

def find_cycles(pi : list) -> list[list]:
    cycles = []
    pos_0 = 0
    while pos_0 < len(pi):
        # Start of current circle
        i = pos_0
        c = [i]
        while(pi[i] != c[0]):
            # We've not come full circle yet.
            i = pi[i]
            c.append(i)
        cycles.append(c)
        # Finding next pos_0 == start of next circle
        pos_found = False
        while (not pos_found):
            pos_0 += 1
            if pos_0 >= len(pi):
                # no more elements in pi, that are not in a circle.
                break
            # Assuming, pos_0 is not in any circle
            pos_found = True
            # If pos_0 in a circle: pos_found = False
            for c in cycles:
                pos_found = pos_found and not (pos_0 in c)
    return cycles
