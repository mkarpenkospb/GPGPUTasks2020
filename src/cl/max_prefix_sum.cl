//#ifdef __CLION_IDE__
//#include <math.h>
//#include "./clion_defines.cl"
//#endif

#line 6
#define WORK_GROUP_SIZE 128



__kernel void max_pref_simple(
        __global const int *prefixIn,
        __global const int *sumIn,
        __global const int *idxIn,
        int n,
        __global int *prefixOut,
        __global int *sumOut,
        __global int *idxOut) {

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);

    __local int local_pref[WORK_GROUP_SIZE];
    __local int local_sum[WORK_GROUP_SIZE];
    __local int local_idx[WORK_GROUP_SIZE];
    local_pref[local_id] = prefixIn[global_id] * (global_id < n);
    local_sum[local_id] = sumIn[global_id] * (global_id < n);
    local_idx[local_id] = idxIn[global_id] * (global_id < n);

    barrier(CLK_LOCAL_MEM_FENCE);
//    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        int maxPref = local_pref[0] > 0 ? local_pref[0] : 0;
        int sum = 0;
        int idx = local_pref[0] > 0 ? local_idx[0] : local_idx[0] - 1;
        int localPref = 0;
        for (unsigned int i = 1; i < WORK_GROUP_SIZE; ++i) {
            sum += local_sum[i - 1];
            localPref = (sum + local_pref[i]);
            if (localPref > maxPref) {
                maxPref = localPref;
                idx = local_idx[i];
            }
        }
        sum += local_sum[WORK_GROUP_SIZE - 1];

        prefixOut[group_id] = maxPref;
        sumOut[group_id] = sum;
        idxOut[group_id] = idx;
    }
}

__kernel void max_pref_using_other_threads(
        __global const int *prefixIn,
        __global const int *sumIn,
        __global const int *idxIn,
        int n,
        __global int *prefixOut,
        __global int *sumOut,
        __global int *idxOut) {

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);

    __local int local_pref[WORK_GROUP_SIZE];
    __local int local_sum[WORK_GROUP_SIZE];
    __local int local_idx[WORK_GROUP_SIZE];

    __local int local_pref_2[WORK_GROUP_SIZE / 2];
    __local int local_sum_2[WORK_GROUP_SIZE / 2];
    __local int local_idx_2[WORK_GROUP_SIZE / 2];


    __local int  * swap;
    // иначе не получился swap
    __local int  * local_pref_pt1 = local_pref;
    __local int  * local_sum_pt1 = local_sum;
    __local int  * local_idx_pt1 = local_idx;

    __local int  * local_pref_pt2 = local_pref_2;
    __local int  * local_sum_pt2 = local_sum_2;
    __local int  * local_idx_pt2 = local_idx_2;

    local_pref[local_id] = (global_id >= n) ? 0 : prefixIn[global_id];
    local_sum[local_id] = (global_id >= n) ? 0 : sumIn[global_id];
    local_idx[local_id] = (global_id >= n) ? 0 : idxIn[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * local_id < nvalues) {
            unsigned int i_1 = 2 * local_id;
            unsigned int i_2 = i_1 + 1;

            local_pref_pt2[local_id] = local_pref_pt1[i_1] < 0 ? 0 :  local_pref_pt1[i_1];
            local_idx_pt2[local_id] = local_pref_pt1[i_1] < 0 ? max(0, local_idx_pt1[i_1] - 1) :  local_idx_pt1[i_1];

            int localPref = local_sum_pt1[i_1] + local_pref_pt1[i_2];

            local_sum_pt2[local_id] = local_sum_pt1[i_1] + local_sum_pt1[i_2];

            if (localPref > local_pref_pt2[local_id]) {
                local_idx_pt2[local_id] = local_idx_pt1[i_2];
                local_pref_pt2[local_id] = localPref;
            }
        }
        // каждый поток должен сделать swap, это было неочевидно
        swap = local_pref_pt1; local_pref_pt1 = local_pref_pt2; local_pref_pt2 = swap;
        swap = local_idx_pt1; local_idx_pt1 = local_idx_pt2; local_idx_pt2 = swap;
        swap = local_sum_pt1; local_sum_pt1 = local_sum_pt2; local_sum_pt2 = swap;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        prefixOut[group_id] = local_pref_pt1[0];
        sumOut[group_id] = local_sum_pt1[0];
        idxOut[group_id] = local_idx_pt1[0];
    }
}

