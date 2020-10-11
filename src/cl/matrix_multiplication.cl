
//#include "./clion_defines.cl"

#define TILE_SIZE 16

__kernel void matrix_multiplication(
        __global const float * as,
        __global const float * bs,
        __global float * cs,
        unsigned int M,
        unsigned int K,
        unsigned int N
        ) {


    unsigned int local_i = get_local_id(1);
    unsigned int local_j = get_local_id(0);

    unsigned int group_i = get_group_id(1);
    unsigned int group_j = get_group_id(0);

    unsigned int global_i = get_global_id(1);
    unsigned int global_j = get_global_id(0);

    __local float a_loc[TILE_SIZE][TILE_SIZE + 1];
    __local float b_loc[TILE_SIZE][TILE_SIZE + 1];
    __local float c_loc[TILE_SIZE][TILE_SIZE + 1];
    c_loc[local_i][local_j] = 0;

    // Мы заранее знаем, что размерность совпадут
    unsigned int loops = M / TILE_SIZE;
    for (int k = 0; k < loops; ++k) {
        int a_i = TILE_SIZE * group_i + local_i; // не зависит от k
        int a_j = TILE_SIZE * k + local_j; // зависит от k

        int b_i = TILE_SIZE * group_j + local_i; // не зависит от k
        int b_j = TILE_SIZE * k + local_j; // зависит от k

        a_loc[local_i][local_j] = as[a_i * K + a_j];
        b_loc[local_i][local_j] = bs[b_i * K + b_j]; // K, так как транспонирована

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i) {
            // после транспонирования строка на строку
            c_loc[local_i][local_j] += a_loc[local_i][i] * b_loc[local_j][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    cs[global_i * N + global_j] = c_loc[local_i][local_j];
}