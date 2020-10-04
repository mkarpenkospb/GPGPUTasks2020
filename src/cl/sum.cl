//#ifdef __CLION_IDE__
//#include <math.h>
//#include "./clion_defines.cl"
//#endif

#define WORK_GROUP_SIZE 256
#define DATA_PER_ITEM 8
// код с лекции
__kernel void sum_tree(__global const unsigned int *data, int n, __global unsigned int* res) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local unsigned int local_data[WORK_GROUP_SIZE];

    local_data[local_id] = data[global_id] * (global_id < n);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * local_id < nvalues) {
            unsigned int a = local_data[local_id];
            unsigned int b = local_data[local_id + nvalues / 2];
            local_data[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(res, local_data[0]);
    }
}


__kernel void sum_tree_less_atomic(__global const unsigned int *data, int n, __global unsigned int* res) {
    int local_id = get_local_id(0);
//    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    __local unsigned int local_data[WORK_GROUP_SIZE];
    unsigned int sum = 0;
    unsigned int group_shift = group_id * WORK_GROUP_SIZE * DATA_PER_ITEM;
    local_data[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < DATA_PER_ITEM; ++i) {
        unsigned int data_id = group_shift + local_id + (i * WORK_GROUP_SIZE);
        local_data[local_id] += data[data_id] * (data_id < n);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * local_id < nvalues) {
            unsigned int a = local_data[local_id];
            unsigned int b = local_data[local_id + nvalues / 2];
            local_data[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        sum += local_data[0];
    }

    if (local_id == 0) {
        atomic_add(res, sum);
    }
}

