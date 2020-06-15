from parameters import *

POS_COMMAND = ".pos(INDEX_VAR,INDEX_VAR_POS,SPARSE_TENSOR)"
SPLIT_COMMAND = ".split(INDEX_VAR,SPLIT_VAR1,SPLIT_VAR2,FACTOR)"
FUSE_COMMAND = ".fuse(INDEX_VAR1,INDEX_VAR2,FUSED_VAR)"
BOUND_COMMAND = ".bound(INDEX_VAR1,INDEX_VAR2,VAL,BoundType::MaxExact)"
PARALLELIZE_COMMAND = ".parallelize(INDEX_VAR,UNIT,REDUCTION)"
PRECOMPUTE_COMMAND = ".precompute(precomputedExpr, SOURCE, DEST, precomputed)"
UNROLL_COMMAND = ".unroll(VAR,VAL)"

def emit_unroll_command(var, val):
    return UNROLL_COMMAND.replace("VAR",var).replace("VAL",val)

def emit_precompute_command(source, dest):
    return PRECOMPUTE_COMMAND.replace("SOURCE",source).replace("DEST",dest)

def emit_split_command(index_var, split_var1, split_var2,factor):
    if factor:
        return SPLIT_COMMAND.replace("INDEX_VAR", index_var).replace("SPLIT_VAR1", split_var1).replace("SPLIT_VAR2",
                                                                                                       split_var2).replace("FACTOR",str(factor))
    else:
        return SPLIT_COMMAND.replace("INDEX_VAR", index_var).replace("SPLIT_VAR1", split_var1).replace("SPLIT_VAR2",
                                                                                                   split_var2)
def emit_bound_command(index_var1, index_var2, value):
    return BOUND_COMMAND.replace("INDEX_VAR1", index_var1).replace("INDEX_VAR2", index_var2).replace("VAL",
                                                                                                    str(value))

def emit_pos_command(index_var, index_var_pos):
    # note the order LOL
    return POS_COMMAND.replace("INDEX_VAR_POS",index_var_pos).replace("INDEX_VAR", index_var).replace("SPARSE_TENSOR",SPARSE_TENSOR)

def emit_fuse_command(index_var1, index_var2, fused_var):
    return FUSE_COMMAND.replace("INDEX_VAR1", index_var1).replace("INDEX_VAR2", index_var2).replace("FUSED_VAR",
                                                                                                    fused_var)

def emit_reorder_command(index_var_list):
    stuff = ".reorder({"
    for var in index_var_list:
        stuff += var + ","
    stuff = stuff[:-1]
    stuff += "})"
    return stuff

def emit_parallelize_command(index_var,unit,race):
    if race == 1:
        return PARALLELIZE_COMMAND.replace("INDEX_VAR",str(index_var)).replace("UNIT",unit).replace("REDUCTION","OutputRaceStrategy::Atomics")
    else:
        return PARALLELIZE_COMMAND.replace("INDEX_VAR",str(index_var)).replace("UNIT",unit).replace("REDUCTION","OutputRaceStrategy::IgnoreRaces")

CPUPAR = "ParallelUnit::CPUThread"
CPUVEC = "ParallelUnit::CPUVector"
GPUBLOCK = "ParallelUnit::GPUBlock"
GPUWARP = "ParallelUnit::GPUWarp"
GPUTHREAD = "ParallelUnit::GPUThread"

def fuse_all_parallelize(index_var_list, dirty_outer, mode):
    if len(index_var_list) == 1:
        if mode == "CPU":
            return [emit_parallelize_command(index_var_list[0], CPUPAR, index_var_list[0] == dirty_outer)]
        elif mode == "GPUBLOCK":
            return [emit_parallelize_command(index_var_list[0], GPUBLOCK, index_var_list[0] == dirty_outer)]
        elif mode == "GPUWARP":
            return [emit_parallelize_command(index_var_list[0], GPUWARP, index_var_list[0] == dirty_outer)]
        elif mode == "GPUALL":
            return [emit_split_command(index_var_list[0], "block", "warp", factor="NUMWARPS"),
                    emit_parallelize_command("block", GPUBLOCK, index_var_list[0] == dirty_outer),
                    emit_parallelize_command("warp", GPUWARP, index_var_list[0] == dirty_outer)
                    ]

    elif len(index_var_list) == 2:
        race = (index_var_list[0] == dirty_outer or index_var_list[1] == dirty_outer)
        if mode == "CPU":
            return [emit_fuse_command(index_var_list[0], index_var_list[1], "f0"),
                    emit_parallelize_command("f0", CPUPAR, race)]
        elif mode == "GPUBLOCK":
            return [emit_fuse_command(index_var_list[0], index_var_list[1], "f0"),
                    emit_parallelize_command("f0", GPUBLOCK, race)]
        elif mode == "GPUWARP":
            return [emit_fuse_command(index_var_list[0], index_var_list[1], "f0"),
                    emit_parallelize_command("f0", GPUWARP, race)]
        elif mode == "GPUALL":
            return [emit_fuse_command(index_var_list[0], index_var_list[1], "f0"),
                    emit_split_command("f0", "block", "warp", factor="NUMWARPS"),
                    emit_parallelize_command("block", GPUBLOCK, index_var_list[0] == dirty_outer),
                    emit_parallelize_command("warp", GPUWARP, index_var_list[0] == dirty_outer)]
    else:
        commands = []
        commands.append(emit_fuse_command(index_var_list[0], index_var_list[1], "f0"))
        race = (index_var_list[0] == dirty_outer or index_var_list[1] == dirty_outer)
        for i in range(2, len(index_var_list)):
            commands.append(emit_fuse_command("f" + str(i - 2), index_var_list[i], "f" + str(i - 1)))
            race = (race or index_var_list[i] == dirty_outer)

        if mode == "CPU":
            commands.append(emit_parallelize_command("f" + str(len(index_var_list) - 2), CPUPAR, race))
        elif mode == "GPUBLOCK":
            commands.append(emit_parallelize_command("f" + str(len(index_var_list) - 2), GPUBLOCK, race))
        elif mode == "GPUWARP":
            commands.append(emit_parallelize_command("f" + str(len(index_var_list) - 2), GPUWARP, race))
        elif mode == "GPUALL":
            commands += [emit_split_command("f" + str(len(index_var_list) - 2), "block", "warp", "NUMWARPS"),
                         emit_parallelize_command("block", GPUBLOCK, index_var_list[0] == dirty_outer),
                         emit_parallelize_command("warp", GPUWARP, index_var_list[0] == dirty_outer)]
        return commands