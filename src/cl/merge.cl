//#include "clion_defines.cl"

#define GROUP_SIZE 256

/*
 * merge_size -- какого размера массивы сливаем, результат 2 * merge_size
 */
__kernel void merge_global_step(__global const float *as, __global float *res, unsigned int merge_size) {
    unsigned int global_id = get_global_id(0);

    unsigned int res_size = 2 * merge_size;
    unsigned int diag_idx = global_id % res_size;
    unsigned int matrix_pos = global_id / res_size;
    unsigned int matrix_start = matrix_pos * res_size;

    unsigned diag_length = (diag_idx >= merge_size ? res_size - diag_idx : diag_idx + 2);
    unsigned r = diag_length;
    unsigned l = 0;
    unsigned int m = 0;
    unsigned int above = 0; // значение сравнения справа сверху
    unsigned int below = 0; // значение сравнения слева снизу

    __global const float *a = as + matrix_start;
    __global const float *b = as + matrix_start + merge_size;

    while (true) {
        m = (l + r) / 2;
        unsigned int below_idx_a = diag_idx >= merge_size ? merge_size - m : diag_length - 1 - m;
        unsigned int below_idx_b = diag_idx >= merge_size ? merge_size - diag_length + m : m - 1;
        unsigned int above_idx_a = below_idx_a - 1;
        unsigned int above_idx_b = below_idx_b + 1;

        below = m == 0 ? 1 : a[below_idx_a] > b[below_idx_b];
        above = m == diag_length - 1 ? 0 : a[above_idx_a] > b[above_idx_b];

        if (below != above) {
            if ((diag_idx < merge_size) && m == 0) {
                res[global_id] = a[above_idx_a];
                return;
            }
            if ((diag_idx < merge_size) && m == diag_length - 1) {
                res[global_id] = b[below_idx_b];
                return;
            }
            // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
            res[global_id] = a[above_idx_a] > b[below_idx_b] ? a[above_idx_a] : b[below_idx_b];
            return;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}

// для первых небольших слияний
__kernel void merge_local_begin(__global float *as) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);

    __local float local_as_data[GROUP_SIZE];
    __local float local_res_data[GROUP_SIZE];
    local_as_data[local_id] = as[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    __local float *local_as = local_as_data;
    __local float *local_res = local_res_data;
    __local float *local_tmp;

    unsigned int merge_size = 1;
    while (merge_size != GROUP_SIZE) {
        unsigned int res_size = 2 * merge_size;
        unsigned int diag_idx = local_id % res_size;
        unsigned int matrix_pos = local_id / res_size;
        unsigned int matrix_start = matrix_pos * res_size;

        unsigned diag_length = (diag_idx >= merge_size ? res_size - diag_idx : diag_idx + 2);
        unsigned r = diag_length;
        unsigned l = 0;
        unsigned int m = 0;
        unsigned int above = 0; // значение сравнения справа сверху
        unsigned int below = 0; // значение сравнения слева снизу

        __local const float *a = local_as + matrix_start;
        __local const float *b = local_as + matrix_start + merge_size;

        while (true) {
            m = (l + r) / 2;
            unsigned int below_idx_a = diag_idx >= merge_size ? merge_size - m : diag_length - 1 - m;
            unsigned int below_idx_b = diag_idx >= merge_size ? merge_size - diag_length + m : m - 1;
            unsigned int above_idx_a = below_idx_a - 1;
            unsigned int above_idx_b = below_idx_b + 1;

            below = m == 0 ? 1 : a[below_idx_a] > b[below_idx_b];
            above = m == diag_length - 1 ? 0 : a[above_idx_a] > b[above_idx_b];

            if (below != above) {
                if ((diag_idx < merge_size) && m == 0) {
                    local_res[local_id] = a[above_idx_a];
                    break;
                }
                if ((diag_idx < merge_size) && m == diag_length - 1) {
                    local_res[local_id] = b[below_idx_b];
                    break;
                }
                // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
                local_res[local_id] = a[above_idx_a] > b[below_idx_b] ? a[above_idx_a] : b[below_idx_b];

                break;
            }
            if (below) {
                l = m;
            } else {
                r = m;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        local_tmp = local_as;
        local_as = local_res;
        local_res = local_tmp;
        merge_size <<= 1;
    }
    as[global_id] = local_as[local_id];
}


