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
        __global int *idxOut
        ) {

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

    if (local_id == 0) {
        int maxPref = local_pref[0] >  0 ? local_pref[0] : 0;
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