from copy import deepcopy
from parameters import *

def no_useless_partition(var_map, filtered_perms,hardware):
    for var in var_map:
        if len(var_map[var]) == 2:
            [outer, inner] = var_map[var]
            if outer not in list(filtered_perms[0]) or inner not in list(filtered_perms[0]):
                continue
            if hardware == "CPU":
                filtered_perms = [perm for perm in filtered_perms if not (
                        list(perm).index(outer) + 1 == list(perm).index(inner) and list(perm).index(outer) > 0)]
            elif hardware == "GPU":
                filtered_perms = [perm for perm in filtered_perms if not (
                        list(perm).index(outer) + 1 == list(perm).index(inner))]
            else:
                raise Exception
    return filtered_perms

def get_concordancy(perm, var_map):

    concordancy = 0
    perm = list(perm)
    local_var_map = deepcopy(var_map)
    token = 0
    order = {}
    for var in perm:
        for var1 in local_var_map:
            if var in local_var_map[var1]:
                local_var_map[var1].remove(var)
                if len(local_var_map[var1]) == 0:
                    order[var1] = token
        token += 1
    for preference in PREFERENCES:
        first, second = preference
        if first not in order or second not in order:
            continue
        if not order[first] > order[second]:
            concordancy += 1

    return concordancy

def enforce_sparse_order(var_map, filtered_perms):
    for compressed_var in PRECEDENCES:
        if var_map[compressed_var] == [compressed_var]:
            compressed_outer = compressed_var
        else:
            compressed_outer = var_map[compressed_var][0]

        """
       This means that somebody has manually moved the compressed variable to the end, probably through a precompute specification
       """
        if compressed_outer not in filtered_perms[0]:
            continue

        for dependent_var in PRECEDENCES[compressed_var]:

            # this suggests that the dependent var and the compresed var were fused, this condition will always be true
            if var_map[dependent_var] == var_map[compressed_var]:
                continue

            if var_map[dependent_var] == [dependent_var]:
                if dependent_var not in filtered_perms[0]:
                    continue
                filtered_perms = [perm for perm in filtered_perms if not
                list(perm).index(compressed_outer) < list(perm).index(dependent_var)]
            else:
                [outer, inner] = var_map[dependent_var]
                if outer not in filtered_perms[0] and inner not in filtered_perms[0]:
                    continue
                elif outer not in filtered_perms[0]:
                    filtered_perms = [perm for perm in filtered_perms if not (list(perm).index(
                        compressed_outer) < list(perm).index(inner))]
                elif inner not in filtered_perms[0]:
                    filtered_perms = [perm for perm in filtered_perms if not (list(perm).index(
                        compressed_outer) < list(perm).index(outer))]
                else:
                    filtered_perms = [perm for perm in filtered_perms if not (
                        list(perm).index(compressed_outer) < list(perm).index(outer) or list(perm).index(
                    compressed_outer) < list(perm).index(
                    inner))]
    return filtered_perms