from parameters import *

def intersect_axis_types(type1,type2):
    if type1 == "Z":
        return type2
    elif type2 == "Z":
        return type1
    elif type1 == "C" or type2 == "C":
        return "C"
    else:
        return "D"

def union_axis_types(type1,type2):
    if type1 == "Z":
        return type2
    elif type2 == "Z":
        return type1
    elif type1 == "D" or type2 == "D":
        return "D"
    else:
        return "C"

def reduce_conjunction(conjunction):
    if len(conjunction) == 1:
        return conjunction[0]
    elif len(conjunction) == 2:
        return intersect_axis_types(conjunction[0],conjunction[1])
    else:
        return reduce_conjunction([intersect_axis_types(conjunction[0],conjunction[1]),conjunction[2:]])

def reduce_disjunction(disjunction):
    if len(disjunction) == 1:
        return disjunction[0]
    elif len(disjunction) == 2:
        return union_axis_types(disjunction[0],disjunction[1])
    else:
        return reduce_disjunction([union_axis_types(disjunction[0],disjunction[1]),disjunction[2:]])

def determine_la_type(this_axis, last_axis, strategy):
    if strategy == "F" or strategy == "Z":
        if last_axis is None:
            return this_axis
        else:
            if this_axis == "C" or last_axis == "C":
                return "C"
            else:
                return "D"
    else:
        return this_axis

import itertools
def findsubsets(s, n):
    return list(itertools.combinations(s, n))

from copy import deepcopy

def alone(value, dictionary, tup):
    for key in dictionary:
        if value in dictionary[key]:
            other_vals = deepcopy(dictionary[key]).remove(value)

    if not other_vals:
        return True

    for val in other_vals:
        if val in tup:
            return False

    return True

def get_dirty_from_schedule(schedule):
    dirty_vars = REDUCTION_AXES.copy()
    dirty = 0
    for command in schedule:
        index_vars = command.split("(")[1].split(")")[0].split(",")
        flag = False
        for var in index_vars:
            if var in REDUCTION_AXES:
                flag = True
        if flag or dirty == 1:
            if "split" in command:
                dirty_vars.append(index_vars[2])
                dirty_vars.append(index_vars[1])
                break
            else:
                dirty = 1



    return dirty_vars


def determine_axis_type(face_matrix):
    # face matrix is a T by 2 numpy array
    assert face_matrix.shape[0] == T

    dnf = []
    for conjunction in DNF:
        clause = []
        for tensor in conjunction:
            clause.append(face_matrix[tensor])
        dnf.append(clause)

    disjunction = []
    for conjunction in dnf:
        disjunction.append(reduce_conjunction(conjunction))
    return reduce_disjunction(disjunction)

def determine_schedule_info(split_schedule):
    inner_split_sizes = {}

    fuse = False
    fused_list = []
    var_map = {}

    for i in range(len(split_schedule)):
        command = split_schedule[i]
        index_vars = command.split("(")[1].split(")")[0].split(",")

        if i < len(split_schedule) - 1:
            next_command = split_schedule[i + 1]
        else:
            next_command = "a"

        if i > 0:
            previous_command = split_schedule[i - 1]
        else:
            previous_command = "a"

        if "pos" in previous_command:
            index_vars[0] = previous_command.split(",")[0].split("(")[1]

        if "split" in command:
            outer = command.split(",")[1] if "bound" not in next_command else next_command.split(",")[1]
            inner = command.split(",")[2]
            factor = command.split(")'")[0].split(",")[-1]
            inner_split_sizes[inner] = factor.replace(")", "")
            if "bound" in next_command:
                inner_split_sizes[outer] = next_command.split(",")[2]

            if not fuse:
                var_map[index_vars[0]] = [outer,inner]
            else:
                for i in fused_list:
                    var_map[i] = [outer,inner]
                fuse = False
                fused_list = []
        elif "fuse" in command:
            if not fuse:
                fused_list.extend([index_vars[1], index_vars[0]])
            else:
                fused_list.append(index_vars[1])
            fuse = True

    for i in INDEX_VARIABLES_0:
        if i not in var_map:
            var_map[i] = [i]

    return inner_split_sizes, var_map

