//#include "./clion_defines.cl"


#define GROUP_SIZE 256

__kernel void radix(__global unsigned int* as,
                    __global unsigned int* res,
                    __global unsigned int* pref_sum_zeroes,
                    __global unsigned int* pref_sum_ones,
                    unsigned int shift,
                    unsigned int n
                    )
{
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int group_num = get_num_groups(0);

    unsigned int total_zeroes = pref_sum_zeroes[n - 1];
    short int val = (as[global_id] >> shift) & 1;
    unsigned int position_for_zero = 0;
    unsigned int position_for_one = total_zeroes;
    if (global_id >= GROUP_SIZE) {
        position_for_zero += pref_sum_zeroes[group_id * GROUP_SIZE - 1];
        position_for_one += pref_sum_ones[group_id * GROUP_SIZE - 1];
    }
    if (local_id != 0) {
        position_for_zero += pref_sum_zeroes[global_id - 1];
        position_for_one += pref_sum_ones[global_id - 1];
    }
    unsigned int position = val ? position_for_one : position_for_zero;
    res[position] = as[global_id];
}


__kernel void pref_sum(__global const unsigned int* as,
                       __global unsigned int* ones_sum,
                       __global unsigned int* zeroes_sum,
                       unsigned int shift) {

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int group_num = get_num_groups(0);

    __local unsigned int tree_zeroes[GROUP_SIZE + GROUP_SIZE / 2];
    __local unsigned int tree_ones[GROUP_SIZE + GROUP_SIZE / 2];

    unsigned int pos = local_id + 1;
    int levels = GROUP_SIZE; //

    unsigned int read_shift = GROUP_SIZE;
    unsigned int write_shift = 0;
    unsigned int tmp = 0;

    unsigned int zeroes = 0;
    unsigned int ones = 0;
    unsigned int tree_size = GROUP_SIZE + GROUP_SIZE / 2;
    while (levels) {
        if (levels == GROUP_SIZE) {
            tree_zeroes[local_id] = ((as[global_id] ^ 0xFFFFFFFF) >> shift) & 1;
            tree_ones[local_id] = (as[global_id] >> shift) & 1;
        }
        else if (local_id < levels) {
            tree_zeroes[write_shift + local_id] =
                    tree_zeroes[read_shift + 2 * local_id] + tree_zeroes[read_shift + 2 * local_id + 1];
            tree_ones[write_shift + local_id] =
                    tree_ones[read_shift + 2 * local_id] + tree_ones[read_shift + 2 * local_id + 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        tmp = write_shift;
        write_shift = read_shift;
        read_shift = tmp;
//        if ((pos & 1) && read_shift + pos - 1 >= tree_size) {
//            printf("SF on 52365275!\n");
//        }
        if (pos & 1) {
            zeroes += tree_zeroes[read_shift + pos - 1];
            ones += tree_ones[read_shift + pos - 1];
        }

        pos >>= 1;
        levels >>= 1;
    }

    zeroes_sum[global_id] = zeroes;
    ones_sum[global_id] = ones;
}


__kernel void count_pref_on_roots(
                          __global unsigned int* in_zeroes,
                          __global unsigned int* in_ones,
                          unsigned int step_between_roots,
                          unsigned int n) {

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int tree_size = GROUP_SIZE + GROUP_SIZE / 2;
    __local unsigned int tree_zeroes[GROUP_SIZE + GROUP_SIZE / 2]; //
    __local unsigned int tree_ones[GROUP_SIZE + GROUP_SIZE / 2];

    unsigned int pos = local_id + 1;
    int levels = GROUP_SIZE;

    unsigned int read_shift = GROUP_SIZE;
    unsigned int write_shift = 0;
    unsigned int tmp = 0;

    unsigned int zeroes = 0;
    unsigned int ones = 0;

    while (levels) {
        if (levels == GROUP_SIZE) {
            if (global_id < n) {
                tree_zeroes[local_id] = in_zeroes[(global_id + 1) * step_between_roots - 1];
                tree_ones[local_id] = in_ones[(global_id + 1) * step_between_roots - 1];
            } else {
                tree_zeroes[local_id] = 0;
                tree_ones[local_id] =  0;
            }
        }
        else if (local_id < levels) {
            tree_zeroes[write_shift + local_id] =
                    tree_zeroes[read_shift + 2 * local_id] + tree_zeroes[read_shift + 2 * local_id + 1];
            tree_ones[write_shift + local_id] =
                    tree_ones[read_shift + 2 * local_id] + tree_ones[read_shift + 2 * local_id + 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        tmp = write_shift;
        write_shift = read_shift;
        read_shift = tmp;

        if (pos & 1) {
            zeroes += tree_zeroes[read_shift + pos - 1];
            ones += tree_ones[read_shift + pos - 1];
        }

        pos >>= 1;
        levels >>= 1;
    }
    if (global_id < n) {
        in_zeroes[(global_id + 1) * step_between_roots - 1] = zeroes;
        in_ones[(global_id + 1) * step_between_roots - 1] = ones;
    }
}

// на самом деле все корни можно хранить прямо в массиве, на котором считаем.
__kernel void update_from_pref(__global unsigned int* in_zeroes,
                                __global unsigned int* in_ones,
                                unsigned int step_between_roots) {
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int real_position = ((global_id + 1) * GROUP_SIZE - 1);
    // global_id / step_between_roots --  в каком отрезке размера step_between_roots мы находимся

    // global_id / step_between_roots  -- в какой группе отрезков (каждая группа по GROUP_SIZE элементов) мы находимся
    unsigned int pref_group = real_position / step_between_roots / GROUP_SIZE;

    // (global_id / step_between_roots) % GROUP_SIZE -- какое место в pref_group мы занимаем
    unsigned int pos = (real_position/ step_between_roots) % GROUP_SIZE;

    // нам не нужно трогать самых левых и сами корни.
    // пример, пусть step_between_roots = 128, тогда global_id == 127 трогать не надо, там корень.
    // если pos == 0, не трогаем!
    if (pos == 0 || (real_position + 1) % (step_between_roots) == 0) {
        return;
    }
    // нас интересуют последние элементы фрагментов, корни, в которых всё посчитано.
    // размер группы -- step_between_roots * GROUP_SIZE
    // нужно попасть на начало группы : pref_group * (step_between_roots * GROUP_SIZE)
    // но на конец отрезка pos : pos * step_between_roots - 1
    // нужно взять значение из корня предыдущего отрезка.

    in_zeroes[real_position] += in_zeroes[pref_group * (step_between_roots * GROUP_SIZE) + pos * step_between_roots - 1];
    in_ones[real_position] += in_ones[pref_group * (step_between_roots * GROUP_SIZE) + pos * step_between_roots - 1];
}