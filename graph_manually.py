from bijective_bwt_st import *

w = "bcbccbcbcabbaaba"
k = 2

st_2 = ST(w, k)

pi = standard_permutation(st_2)

m_2 = M(w, k)

print("\n".join(m_2))