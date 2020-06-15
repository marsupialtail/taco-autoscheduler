
myfunction = """
    return stmt.reorder({k,i,j})
            .split(i,x0,x1,FACTOR0)
            .pos(j,x2,A(i,j))
            .split(x2,x3,x4,FACTOR1)
            .split(k,x5,thread,32)
            .bound(x5,x6,4,BoundType::MaxExact)
            .reorder({ORDER})
            .parallelize(BLOCK,ParallelUnit::GPUBlock,OutputRaceStrategy::IgnoreRaces)
            .parallelize(WARP,ParallelUnit::GPUWarp,OutputRaceStrategy::IgnoreRaces)
            .parallelize(thread,ParallelUnit::GPUThread,OutputRaceStrategy::IgnoreRaces);

"""


def is_concordant(perm):
    perm = list(perm)
    i = max(perm.index('x1'),perm.index('x0'))
    j = max(perm.index('x3'),perm.index('x4'))
    k = max(perm.index('x5'),perm.index('x6'))
    return (i < j and j < k and i < k)
from itertools import permutations
variables = ['x0','x1','x3','x4','x6']
par_vars = ['x0','x1','x4','x6']

mappings = []
for block_var in par_vars:
    for warp_var in par_vars:
        if block_var == warp_var or ('x0' in {block_var,warp_var} and 'x1' in {block_var,warp_var}):
            continue
        remainder_vars = [i for i in variables if i is not block_var and i is not warp_var]
        filtered_perms = list(permutations(remainder_vars))
        if 'x0' in remainder_vars:
            filtered_perms = [perm for perm in filtered_perms if list(perm).index('x3') > list(perm).index('x0')]
        if 'x1' in remainder_vars:
            filtered_perms = [perm for perm in filtered_perms if list(perm).index('x3') > list(perm).index('x1')]
        if 'x3' in remainder_vars and 'x4' in remainder_vars:
            filtered_perms = [perm for perm in filtered_perms if not list(perm).index('x3') + 1 == list(perm).index('x4')]
        if 'x0' in remainder_vars and 'x1' in remainder_vars:
            filtered_perms = [perm for perm in filtered_perms if
                          not (list(perm).index('x0') + 1 == list(perm).index('x1'))]
        mappings.extend([(a, b) for (a, b) in zip(filtered_perms, [(block_var,warp_var)] * len(filtered_perms))])

print(len(mappings))
for mapping in mappings:
    print(mapping)

numcases = len(mappings)

define = """

"""
for case in range(numcases):
    mapping = mappings[case]
    define += "if(num == " + str(case) + "){"
    block_var = mapping[1][0]
    warp_var = mapping[1][1]
    permutation = [block_var,warp_var] + list(mapping[0]) + ["thread"]

    func = myfunction.replace("ORDER",",".join(permutation)).replace("BLOCK",block_var).replace("WARP",warp_var)
    if block_var == "x4" or warp_var == "x4":
        func = func.replace("GPUThread,OutputRaceStrategy::IgnoreRaces","GPUThread,OutputRaceStrategy::Atomics")

    define += func  + "\n"
    define += "}\n"


print(define)