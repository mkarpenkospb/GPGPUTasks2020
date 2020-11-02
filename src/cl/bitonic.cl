//#include "./clion_defines.cl"

#define GROUP_SIZE 256

//void swap(unsigned int* a, unsigned int* b) {
//    unsigned int
//}

__kernel void local_bitonic_begin(__global float* as) {
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int work_size = GROUP_SIZE * 2;

    __local float local_as[GROUP_SIZE * 2];
    float tmp = 0;
    local_as[local_id] = as[work_size * group_id + local_id];
    local_as[local_id + GROUP_SIZE] = as[work_size * group_id + local_id + GROUP_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int outer = work_size;
    unsigned int segment_length = 2;
    while (outer != 1) {
        unsigned int local_line_id = local_id % (segment_length / 2);
        unsigned int local_twin_id = segment_length - local_line_id - 1;
        unsigned int group_line_id = local_id / (segment_length / 2);
        unsigned int line_id = segment_length * group_line_id + local_line_id;
        unsigned int twin_id = segment_length * group_line_id + local_twin_id;

        if (local_as[line_id] > local_as[twin_id]) {
            tmp = local_as[line_id];
            local_as[line_id] = local_as[twin_id];
            local_as[twin_id] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int j = segment_length / 2; j > 1; j >>= 1) {
            local_line_id = local_id % (j / 2);
            local_twin_id = local_line_id + (j / 2);
            group_line_id = local_id / (j / 2);
            line_id = j * group_line_id + local_line_id;
            twin_id = j * group_line_id + local_twin_id;
            if (local_as[line_id] > local_as[twin_id]) {
                tmp = local_as[line_id];
                local_as[line_id] = local_as[twin_id];
                local_as[twin_id] = tmp;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        outer >>= 1;
        segment_length <<= 1;
    }

    as[work_size * group_id + local_id] =  local_as[local_id];
    as[work_size * group_id + local_id + GROUP_SIZE] = local_as[local_id + GROUP_SIZE];
}

__kernel void bitonic_global_step(__global float* as, unsigned int segment_length, unsigned int mirror)
{
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_line_id = global_id % (segment_length / 2);
    unsigned int local_twin_id = mirror ? segment_length - local_line_id - 1 : local_line_id + (segment_length / 2);
    unsigned int group_line_id = global_id / (segment_length / 2);
    unsigned int line_id = segment_length * group_line_id + local_line_id;
    unsigned int twin_id = segment_length * group_line_id + local_twin_id;

    float tmp = 0;

    if (as[line_id] > as[twin_id]) {
        tmp = as[line_id];
        as[line_id] = as[twin_id];
        as[twin_id] = tmp;
    }
}

__kernel void bitonic_local_endings(__global float* as)
{
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int work_size = GROUP_SIZE * 2;

    __local float local_as[GROUP_SIZE * 2];
    float tmp = 0;

    local_as[local_id] = as[work_size * group_id + local_id];
    local_as[local_id + GROUP_SIZE] = as[work_size * group_id + local_id + GROUP_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int segment_length = work_size;

    for (unsigned int j = segment_length; j > 1; j >>= 1) {
        unsigned int local_line_id = local_id % (j / 2);
        unsigned int local_twin_id = local_line_id + (j / 2);
        unsigned int group_line_id = local_id / (j / 2);
        unsigned int line_id = j * group_line_id + local_line_id;
        unsigned int twin_id = j * group_line_id + local_twin_id;

        if (local_as[line_id] > local_as[twin_id]) {
            tmp = local_as[line_id];
            local_as[line_id] = local_as[twin_id];
            local_as[twin_id] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    as[work_size * group_id + local_id] =  local_as[local_id];
    as[work_size * group_id + local_id + GROUP_SIZE] = local_as[local_id + GROUP_SIZE];
}
