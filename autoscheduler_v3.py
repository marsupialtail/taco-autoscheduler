"""

Hopefully, this is the final version of the autoscheduler

"""

from commands import *
from utils import *
from trimpasses import *
import sys
import argparse

"""
Basically, the core autoscheduler generates a scheduling command sequence. We need some helper code to generate the test
harness around this scheduling command sequence, set up Taco tensors and check results and print generated code etc.
This is what those code_blocks_* include files are. They are pretty similar to each other. A better software engineer would 
probably only include one code blocks file and use some complicated switch case etc. in that file to reduce code duplication.
But this design is better for prototyping if you want to do different things for different tensor problems.
"""

if HARDWARE == "CPU":
    if PROBLEM == "MTTKRP":
        from code_blocks_mttkrp_cpu import *
    elif PROBLEM == "SPMV":
        from code_blocks_spmv import *
    elif PROBLEM == "SPMM":
        from code_blocks_spmm import *
    else:
        print("Warning, no corresponding code blocks, code incorrect")
        from code_blocks_spmm import *
elif HARDWARE == "GPU":
    if PROBLEM == "MTTKRP":
        from code_blocks_mttkrp_gpu import *
    elif PROBLEM == "SPMV":
        from code_blocks_spmv_gpu import *
    elif PROBLEM == "SPMM":
        from code_blocks_spmm_gpu import *
    elif PROBLEM == "SPMSPV":
        from code_blocks_spmspv_gpu import *
    else:
        print("Warning, no corresponding code blocks, code incorrect")
        from code_blocks_spmm_gpu import *

parser = argparse.ArgumentParser(description='Autoscheduler V3')

def process_split(strategy, last_axis_name, this_axis_name, curr_pos, curr_factor):
    """
    takes as input a split strategy, such as "CC" and generate the corresponding commands.
    :param strategy: what's the splitting strategy? "CC"? "PP"?
    :param last_axis_name
    :param this_axis_name
    :param curr_pos: this is needed to name the iteration variables appropriately
    :param curr_factor: this is needed to name the split factors appropriately
    :return: [commands, schedulable_index_vars, next_pos, new_last_axis_name, curr_factor]
    commands is a list of scheduling commands that can be appended to other commands
    , scheduleable_index_var are the new index variables generated in these scheduling commands,
    next_pos and new_last_axis_name self explanatory to feed to next invocation of process_split
    curr_factor also self explanatory.
    """
    axis0 = "x" + str(curr_pos)
    axis1 = "x" + str(curr_pos + 1)
    axis2 = "x" + str(curr_pos + 2)
    axis3 = "x" + str(curr_pos + 3)

    if strategy == "C":
        if this_axis_name in KNOWN_AXES:
            commands = [emit_split_command(this_axis_name, axis0, axis1, factor="FACTOR" + str(curr_factor)),
                        emit_bound_command(axis0, axis2,
                                           str(KNOWN_AXES[this_axis_name]) + "/FACTOR" + str(curr_factor))]
            next_pos = curr_pos + 3
            schedulable_index_vars = [axis2, axis1]

        else:
            commands = [emit_split_command(this_axis_name, axis0, axis1, factor="FACTOR" + str(curr_factor))]
            next_pos = curr_pos + 2
            schedulable_index_vars = [axis0, axis1]

        new_last_axis_name = axis1
        curr_factor += 1

    elif strategy == "Z":
        commands = []
        schedulable_index_vars = [this_axis_name]
        next_pos = curr_pos
        new_last_axis_name = this_axis_name

    elif strategy == "PS":
        commands = [emit_pos_command(this_axis_name, axis0),
                    emit_split_command(axis0, axis1, axis2, factor="FACTOR" + str(curr_factor))]
        schedulable_index_vars = [axis1, axis2]
        next_pos = curr_pos + 3
        new_last_axis_name = axis2
        curr_factor += 1
        # if this is a sparse loop and you split it in position space, the outer variable length is not fixed
        # meaning that it cannot effectively be used for tiling purposes. Although maybe this should be an
        # outer variable if this is the outer most loop.

    elif strategy == "FPS":
        commands = [
            emit_fuse_command(last_axis_name, this_axis_name, axis0),
            emit_pos_command(axis0, axis1),
            emit_split_command(axis1, axis2, axis3, factor="FACTOR" + str(curr_factor))
        ]
        schedulable_index_vars = [axis2, axis3]
        next_pos = curr_pos + 4
        new_last_axis_name = axis3
        curr_factor += 1

        # same reasoning as above

    elif strategy == "F":
        commands = [
            emit_fuse_command(last_axis_name, this_axis_name, axis0)
        ]
        schedulable_index_vars = [axis0]
        next_pos = curr_pos + 1
        new_last_axis_name = axis0
    else:
        raise Exception("strategy not implemented")

    return commands, schedulable_index_vars, next_pos, new_last_axis_name, curr_factor

def emit_split_schedules(schedules, schedule, schedule_vars, problem_matrix, index_variables,
                         curr_pos, last_axis_type, last_axis_fusable, last_axis_name, curr_factor):
    if index_variables.shape[0] == 0:
        # print(schedule)

        dirty_vars = get_dirty_from_schedule(schedule)
        inner_split_sizes, var_map = determine_schedule_info(schedule)

        schedules.append((schedule , schedule_vars, dirty_vars, var_map, inner_split_sizes))

        return

    axis_name = index_variables[0]
    axis_type = determine_axis_type(problem_matrix[:, 0])

    if problem_matrix.shape[1] == N:
        strategies_dict = STRATEGIES_0[axis_type]
        assert last_axis_fusable == None
        assert last_axis_type == None
        assert last_axis_name == None
    else:
        if len(schedule) > 0 and "fuse" in schedule[-1]:
            strategies_dict = STRATEGIES_2[axis_type]
        else:
            strategies_dict = STRATEGIES_1[(axis_type, last_axis_fusable)]

    for strategy in strategies_dict:
        new_last_axis_fusable = LA_FUSABILITY[strategy]
        new_last_axis_type = determine_la_type(axis_type, last_axis_type, strategy)
        commands, schedulable_index_vars,  next_pos, new_last_axis_name, new_factor = process_split(
            strategy, last_axis_name,
            axis_name, curr_pos, curr_factor)
        new_problem_matrix = problem_matrix.copy()[:, 1:]
        new_index_variables = index_variables.copy()[1:]


        if strategy == "F" or strategy == "FPS":
            emit_split_schedules(schedules, schedule + commands,
                                 schedule_vars[:-1] + schedulable_index_vars,
                                 new_problem_matrix, new_index_variables, next_pos, new_last_axis_type,
                                 new_last_axis_fusable, new_last_axis_name, new_factor)
        else:
            emit_split_schedules(schedules, schedule + commands, schedule_vars + schedulable_index_vars,
                                 new_problem_matrix, new_index_variables, next_pos, new_last_axis_type,
                                 new_last_axis_fusable, new_last_axis_name, new_factor)

"""
Input: split schedule tuple, split schedule, dirty vars, var_map, inner_split_sizes
Output: list of tuples of schedule (split schedule + reorder command), what to parallelize

"""

from itertools import permutations

def emit_reorder_schedules(schedule, hardware):

    split_schedule, schedule_vars, dirty_vars, var_map, inner_sizes = schedule
    #print(dirty_vars)

    if MANUAL_PRECOMPUTE:
        for var in MANUAL_PRECOMPUTED_VARS:
            for var1 in var_map[var]:
                schedule_vars.remove(var1)
        if len(schedule_vars) == 0:
            return []

    if hardware == "CPU":
        filtered_perms = list(permutations(schedule_vars))

        """
        Trim pass CPU Sparse
        """

        filtered_perms = enforce_sparse_order(var_map, filtered_perms)
        """
        Trim pass CPU No Useless Partition
        """
        filtered_perms = no_useless_partition(var_map,filtered_perms,hardware)

        """
        Trim pass Concordant Iteration
        """
        concordancy = [get_concordancy(perm,var_map) for perm in filtered_perms]
        desired_concordancy = np.max(concordancy)
        filtered_perms = [x for (i,x) in zip(range(len(filtered_perms)),filtered_perms) if concordancy[i] == desired_concordancy]

        assignment_schedules = []
        for perm in filtered_perms:

            perm = list(perm)

            """
            Trim pass CPU Vectorize
            """
            if MANUAL_PRECOMPUTE:
                """
                Currently in TACO precomputed variables are not supported to be vectorized!!
                """
                # if len(MANUAL_PRECOMPUTED_VARS) == 1:
                #     vec_par_vars = [MANUAL_PRECOMPUTED_VARS[0]]
                # else:
                #     vec_par_vars = MANUAL_PRECOMPUTED_VARS[-2:]
                vec_par_vars = []
            else:
                if len(perm) == 1:
                    vec_par_vars = []
                elif len(perm) == 2:
                    vec_par_vars = [perm[-1]]
                else:
                    vec_par_vars = perm[-2:]

            """
           Trim pass CPU Parallelize Outer
           """

            thread_var = perm[0]
            assignment_schedule = split_schedule + [emit_reorder_command(perm), emit_parallelize_command(thread_var,CPUPAR, thread_var in dirty_vars)]
            assignment_schedules.append(assignment_schedule)
            for vec_var in vec_par_vars:
                if vec_var != thread_var and (vec_var in inner_sizes or vec_var in INDEX_VARIABLES_0):
                    assignment_schedules.append(assignment_schedule + [emit_parallelize_command(vec_var, CPUVEC, vec_var in dirty_vars)])
        return assignment_schedules

    elif hardware == "GPU":

        parallelizable_vars = schedule_vars.copy()
        for compressed_var in PRECEDENCES:
            """
            Even if one of the dependent variables are not fused with this compressed variable, we cannot parallelize this variable's outer split
            """
            for depedent_var in PRECEDENCES[compressed_var]:
                if var_map[depedent_var] != var_map[compressed_var]:
                    """
                    This will either be the outer split for split, or the same if no split
                    """
                    if var_map[compressed_var][0] in parallelizable_vars:
                        parallelizable_vars.remove(var_map[compressed_var][0])

                    """
                    Trim pass GPU: Enforce load balance
                    """

                    for var in var_map[depedent_var]:

                        if var in parallelizable_vars:
                            parallelizable_vars.remove(var)
                #todo: for divide, this will be var_map[compressed_var][1]


        """
        We are going to treat the warp as the smallest unit.
        First strategy, parallelize one variable to the end. This variable could be of any kind, other fixed total or an inner split variable
        """

        mappings = []
        for par_var in parallelizable_vars:

            """
            Trim pass GPU: Enough parallelism
            """
            toosmall = False
            for index_var in var_map:
                if par_var in var_map[index_var] and index_var in KNOWN_AXES and int(KNOWN_AXES[index_var]) < 64:
                    toosmall = True
            if toosmall:
                continue

            remainder_vars = [i for i in schedule_vars if i is not par_var]
            filtered_perms = list(permutations(remainder_vars))

            """
            Trim pass GPU: No Useless Partition
            """
            filtered_perms = no_useless_partition(var_map,filtered_perms,hardware)

            """
            Trim pass GPU: Sparse Ordering
            """
            filtered_perms = enforce_sparse_order(var_map, filtered_perms)

            """
            Trim pass GPU: Concordant
            """
            concordancy = [get_concordancy([par_var] + list(perm), var_map) for perm in filtered_perms]
            desired_concordancy = np.max(concordancy)
            filtered_perms = [x for (i, x) in zip(range(len(filtered_perms)), filtered_perms) if
                              concordancy[i] == desired_concordancy]

            mappings.extend([(a, b) for (a, b) in zip(filtered_perms, [(par_var,par_var)] * len(filtered_perms))])

        """
        Second strategy, parallelize one variable over blocks and another over warps
        """

        for par_var in parallelizable_vars:
            """
           Trim pass GPU: Determinate warp size
           """
            if par_var not in inner_sizes:
                continue

            for block_var in [x for x in parallelizable_vars if x is not par_var]:

                """
                We can't parallelize the two variables resulting from the same index variable!
                """
                escape = False
                for var in var_map:
                    if not(len(var_map[var]) ==2):
                        continue
                    if not(var_map[var][0] == par_var or var_map[var][1] == par_var):
                        continue
                    if len(var_map[var]) == 2 and var_map[var][0] == block_var or var_map[var][1] == block_var:
                        escape = True
                        break

                if escape:
                    continue

                """
                Trim pass GPU: Enough parallelism
                """
                toosmall = False
                for index_var in var_map:
                    if block_var in var_map[index_var] and index_var in KNOWN_AXES and int(KNOWN_AXES[index_var]) < 64:
                        toosmall = True
                if toosmall:
                    continue

                remainder_vars = [i for i in schedule_vars if i is not par_var and i is not block_var]
                filtered_perms = list(permutations(remainder_vars))

                """
                Trim pass GPU: NO Useless Partition
                """
                filtered_perms = no_useless_partition(var_map, filtered_perms,hardware)

                """
                Trim pass GPU: Sparse Order
                """
                filtered_perms = enforce_sparse_order(var_map, filtered_perms)

                """
                Trim pass GPU: Concordant
                """
                concordancy = [get_concordancy([block_var, par_var] + list(perm), var_map) for perm in filtered_perms]
                desired_concordancy = np.max(concordancy)
                filtered_perms = [x for (i, x) in zip(range(len(filtered_perms)), filtered_perms) if
                                  concordancy[i] == desired_concordancy]

                mappings.extend([(a, b) for (a, b) in zip(filtered_perms, [(block_var,par_var)] * len(filtered_perms))])

        """
        starting printing schedules
        
        """
        assignment_schedules = []
        for mapping in mappings:

            permutation = list(mapping[0])
            par_vars = mapping[1]
            if par_vars[0] == par_vars[1]:
                par_var = par_vars[0]
                dirty = par_var if (par_var in dirty_vars) else "a"
                assignment_schedule = split_schedule + [emit_reorder_command([par_var] + permutation)] + fuse_all_parallelize([par_var],dirty,"GPUALL")
            else:
                block_var, warp_var = par_vars
                assignment_schedule = split_schedule + \
                                      [emit_reorder_command([block_var, warp_var] + permutation),
                                       emit_parallelize_command(block_var,GPUBLOCK,block_var in dirty_vars),
                                       emit_parallelize_command(warp_var,GPUWARP,block_var in dirty_vars)]

            """
            If the other split variable is also not in the permutation, otherwise no point
            """
            thread_par_vars = [var for var in permutation if alone(var,var_map,permutation) and (var in inner_sizes or var in INDEX_VARIABLES_0)]
            if len(thread_par_vars) == 0:
                continue

            for thread_var in thread_par_vars:
                dirty = (thread_var in dirty_vars)
                inner_sizes.update(KNOWN_AXES)
                if thread_var in inner_sizes:
                    if PRECOMPUTE_REDUCTION:

                        assignment_schedules.append(
                            assignment_schedule + [emit_split_command(thread_var, "thread", "thread_nz",
                                                                      factor=inner_sizes[thread_var] + "/32"),
                                                   emit_precompute_command("thread_nz","thread_nz_pre"),
                                                   emit_unroll_command("thread_nz_pre",inner_sizes[thread_var] + "/32"),
                                                   emit_parallelize_command("thread", GPUTHREAD, dirty)])
                        """
                        Unfortunately you cannot emit a reorder command here. Fortunately you don't need to. LOL. 
                        """
                    else:
                        assignment_schedules.append(assignment_schedule + [emit_split_command(thread_var, "thread", "thread_nz",
                                                                    factor=inner_sizes[thread_var] + "/32"),
                                                 emit_parallelize_command("thread", GPUTHREAD, dirty),
                                                                           emit_reorder_command(
                                                                               ["block", "warp", "thread"] + [
                                                                                   var if var != thread_var else "thread_nz"
                                                                                   for var in permutation])]
                                                    )

                    assignment_schedules.append(
                        assignment_schedule + [emit_split_command(thread_var, "thread_nz", "thread", factor=32),
                                    emit_bound_command("thread_nz", "thread_nz_bounded",
                                                       inner_sizes[thread_var] + "/32"),
                                    emit_parallelize_command("thread", GPUTHREAD, dirty),
                    emit_reorder_command(["block","warp","thread"] + [var if var!= thread_var else "thread_nz_bounded" for var in permutation])])
                else:
                    assignment_schedules.append(
                        assignment_schedule + [emit_split_command(thread_var, "thread_nz", "thread", factor=32),
                                    emit_parallelize_command("thread", GPUTHREAD, dirty),
                                               emit_reorder_command(["block","warp","thread"] + [var if var!= thread_var else "thread_nz" for var in permutation])])

        return assignment_schedules
        #return mappings


def generate_code(reordered_schedules, hardware, num, factors):

    """
    locally expand num warps
    """

    new_reordered_schedules = []
    for schedule in reordered_schedules:
        if "NUMWARPS" in "".join(schedule):
            #for num_warps in [8, 16, 32]:
            for num_warps in [16]:
                new_reordered_schedules.append([i.replace("NUMWARPS",str(num_warps)) for i in schedule])
        else:
            new_reordered_schedules.append(schedule)

    reordered_schedules = new_reordered_schedules
    numcases = len(new_reordered_schedules)

    if PRECOMPUTE_REDUCTION:
        define = """
        TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);
        """
    else:
        define = """

        """
    for case in range(numcases):

        schedule = reordered_schedules[case]

        """
           Code generation passes
        """

        # pass 1: reorder parallelize to be last commands

        schedule = [i for i in schedule if "parallelize" not in i] + [i for i in schedule if "parallelize" in i]

        # pass 3: move all GPU atomics to the end
        if hardware == "GPU":
            gpu_atomics = False
            for command in schedule:
                if "GPU" in command and "Atomics" in command:
                    gpu_atomics = True

            schedule = [i.replace("GPUWarp,OutputRaceStrategy::Atomics",
                                  "GPUWarp,OutputRaceStrategy::IgnoreRaces") for i in schedule]
            schedule = [i.replace("GPUBlock,OutputRaceStrategy::Atomics",
                                  "GPUBlock,OutputRaceStrategy::IgnoreRaces") for i in schedule]
            if gpu_atomics:
                schedule = [i.replace("GPUThread,OutputRaceStrategy::IgnoreRaces",
                                                                  "GPUThread,OutputRaceStrategy::Atomics") for i in schedule]

        """
        generate some guards to prevent floating point exception due to bad parameters
        this is because some reordered schedules of the same split schedule are mutually exclusive in terms of validity
        """
        guard = """
        if(FACTOR == 0) 
        {throw 2;}
        """
        define += "if(num == " + str(case) + "){\n"

        for command in schedule:
            if "FACTOR" not in command:
                continue
            stuff = command.split(",")
            factor = [i.replace(")","") for i in stuff if "FACTOR" in i][0]
            define += guard.replace("FACTOR",factor)

        #define += "return stmt.reorder({INITIAL})\n\t\t"
        define += "return stmt\n\t\t"

        for command in schedule:
            define += command + "\n\t\t"
        define += ";}\n\n\t"

    factors_def = """
    """
    for i in range(len(factors)):
        factors_def += "#define FACTOR" + str(i) + " " + str(factors[i]) + "\n"

    filename = "test_" + str(num) + "_" + "_".join([str(i) for i in factors]) + ".cpp"
    file = BEGIN.replace("DEFINEBLOCK",factors_def) + define.replace("INITIAL",",".join(INDEX_VARIABLES_0)) + END.replace("NUMSCHEDULES",str(numcases)).replace("BASE",str(filename.split(".")[0]))
    open(filename,"w").write(file)
    return filename


def get_factor_vals(num_factors):
    factor_vals = FACTORS
    if num_factors == 2:
        factor_vals = list(itertools.product(factor_vals, FACTORS))
    elif num_factors > 2:
        for i in range(num_factors - 1):
            factor_vals = list(itertools.product(factor_vals, FACTORS))
            if i > 0:
                factor_vals = [[j for j in i[0]] + [i[1]] for i in factor_vals]
    else:
        factor_vals = [[i] for i in factor_vals]
    return factor_vals


import itertools

split_schedules = []

emit_split_schedules(split_schedules, [], [], PROBLEM_MATRIX_0, INDEX_VARIABLES_0, 0,
                     None, None, None, 0)

"""
Passes to trim split schedules
"""
split_schedules = [x for x in split_schedules if len(x[0]) == 0 or "fuse" not in x[0][-1]]

import os

num = 0

all_schedule_files = []

from math import factorial
def f(n):
    if HARDWARE == "GPU":
        return factorial(n) * (3 * n -4)
    else:
        return factorial(n) * n * n

for schedule in split_schedules:

    flag = False
    if PRECOMPUTE_VAR:
        for command in schedule[0]:
            if "split" in command and PRECOMPUTE_VAR in command:
                flag = True
                break
    if flag:
        continue

    schedule_files = []

    """
    Each split schedule will have one big file containing all the reordered schedules
    """

    print(num)
    print(schedule)
    print(f(len(schedule[1])))
    reordered_schedules = emit_reorder_schedules(schedule,HARDWARE)
    numcases = len(reordered_schedules)
    print(numcases)
    if numcases == 0:
        num += 1
        continue

    """
    Use the improved new way of making the generated code. 

    All the reorderings will be in the same file, corresponding to the same partition. This is because then they will all need the same number of split factors.
    Will need to think more about GPU schedules. 

    """

    num_factors = len([i for i in schedule[0] if "FACTOR" in i and "bound" not in i and "thread" not in i])
    #print(num_factors)

    factor_vals = get_factor_vals(num_factors)

    for factor_val in factor_vals:
        schedule_file = generate_code(reordered_schedules, hardware=HARDWARE, num = num, factors = factor_val)
        schedule_files.append(schedule_file)

    num += 1

    all_schedule_files.append(schedule_files)











