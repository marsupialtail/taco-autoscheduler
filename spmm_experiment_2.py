myfunction = """
    return stmt.reorder({i,j,k})
            .split(i,x0,x1,FACTOR0)
            .pos(j,x2,A(i,j))
            .split(x2,x3,x4,FACTOR1)
            .split(k,x5,x6,FACTOR2)
            .reorder({ORDER})
            .parallelize(PARVAR,ParallelUnit::CPUThread,RACE);
            //.parallelize(x3,ParallelUnit::CPUVector,OutputRaceStrategy::IgnoreRaces)
"""
def is_concordant(perm):
    perm = list(perm)
    i = max(perm.index('x1'),perm.index('x0'))
    j = max(perm.index('x3'),perm.index('x4'))
    k = max(perm.index('x5'),perm.index('x6'))
    return (i < j and j < k and i < k)
from itertools import permutations
variables = ['x0','x1','x3','x4','x5','x6']
filtered_perms = list(permutations(variables))
print(len(filtered_perms))
filtered_perms = [perm for perm in filtered_perms if list(perm).index('x3') > list(perm).index('x0') and list(perm).index('x3') > list(perm).index('x1')]
filtered_perms = [perm for perm in filtered_perms if not (list(perm).index('x3') +1 == list(perm).index('x4') and list(perm).index('x3') > 0)]
filtered_perms = [perm for perm in filtered_perms if not (list(perm).index('x0') +1 == list(perm).index('x1') and list(perm).index('x0') > 0)]
filtered_perms = [perm for perm in filtered_perms if not (list(perm).index('x5') +1 == list(perm).index('x6') and list(perm).index('x5') > 0)]
filtered_perms = [perm for perm in filtered_perms if is_concordant(perm)]




print(len(filtered_perms))
for perm in filtered_perms:
    print(str(perm))

numcases = len(filtered_perms)

define = """

"""
for case in range(numcases):
    define += "if(num == " + str(case) + "){"
    par_var = filtered_perms[case][0]
    func = myfunction.replace("ORDER",",".join(filtered_perms[case])).replace("PARVAR",par_var)
    if par_var == "x4":
        func = func.replace("RACE","OutputRaceStrategy::Atomics")
    else:
        func = func.replace("RACE","OutputRaceStrategy::IgnoreRaces")
    define += func  + "\n"
    define += "}\n"


print(define)