
//#include "./clion_defines.cl"

#define TILE_SIZE 16

__kernel void matrix_transpose(
        __global const float * as,
        __global float * as_t,
        unsigned int M, unsigned int K) {

    unsigned int group_id_i = get_group_id(1);
    unsigned int group_id_j = get_group_id(0);

    unsigned int local_i = get_local_id(1);
    unsigned int local_j = get_local_id(0);

    // twin -- элемент, которым мы прикидываемся при записи
    unsigned int local_twin_i = local_j;
    unsigned int local_twin_j = local_i;

    unsigned int global_i = get_global_id(1);
    unsigned int global_j = get_global_id(0);

    unsigned int global_twin_i = group_id_i * TILE_SIZE + local_twin_i;
    unsigned int global_twin_j = group_id_j * TILE_SIZE + local_twin_j;

    __local float local_data[TILE_SIZE][TILE_SIZE + 1];

    // 1024 делится на всё на свете, в общем случае удобно сделать padding
    local_data[local_j][local_i] = as[global_i * K + global_j];
    barrier(CLK_LOCAL_MEM_FENCE);
    as_t[global_twin_j * M + global_twin_i] = local_data[local_i][local_j];
}