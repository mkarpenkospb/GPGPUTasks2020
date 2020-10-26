//#include "./clion_defines.cl"


#define GROUP_SIZE 256

__kernel void radix(__global unsigned int* as,
                    __global unsigned int* res,
                    __global unsigned int* pref_sum_local_zeroes,
                    __global unsigned int* pref_sum_local_ones,
                    __global unsigned int* pref_sum_global_zeroes,
                    __global unsigned int* pref_sum_global_ones,
                    unsigned int shift,
                    unsigned int n
                    )
{
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int group_num = get_num_groups(0);

    unsigned int total_zeroes = pref_sum_global_zeroes[n / GROUP_SIZE - 1];
//    if (global_id == 34563) {
//        printf("here shift: %d\n", shift);
//        printf("global_id: %d\n", global_id);
//        printf("group_id: %d\n", group_id);
//        printf("zeroes[0]: %d\n", pref_sum_global_zeroes[0]);
//        printf("ones[0]: %d\n", pref_sum_global_ones[0]);
//    }
//    if (global_id == 34563) {
//        printf("total_zeroes: %d\n", total_zeroes);
//    }
    short int val = (as[global_id] >> shift) & 1;
//    if (global_id == 34563) {
//        printf("val: %d\n", val);
//    }
    unsigned int zeroes_before = global_id >= GROUP_SIZE  ? pref_sum_global_zeroes[group_id - 1] : 0;
//    if (global_id == 34563) {
//        printf("zeroes_before: %d\n", zeroes_before);
//    }
    unsigned int ones_before = global_id >= GROUP_SIZE  ?  pref_sum_global_ones[group_id - 1] : 0;
//    if (global_id == 34563) {
//        printf("ones_before: %d\n", ones_before);
//    }
    unsigned int position_for_zero = pref_sum_local_zeroes[global_id] + zeroes_before - 1;
//    if (global_id == 34563) {
//        printf("position_for_zero: %d\n", position_for_zero);
//    }
    unsigned int position_for_one = pref_sum_local_ones[global_id] + ones_before - 1;
//    if (global_id == 34563) {
//        printf("position for one: %d\n", position_for_one);
//    }
    position_for_one += total_zeroes;
//    if (global_id == 34563) {
//        printf("position_for_one: %d\n", position_for_one);
//    }
    unsigned int position = val ? position_for_one : position_for_zero;
//    if (position > 1048576 && local_id == 30) {
////        printf("position large: %d\n", position);
//        printf("--------\nlocal_id: %d,\n group_id: %d,\n total_zeros %d,\n zeroes_before %d,\n ones_before %d,\n zeroes_local: %d,\n ones_local: %d,\n valReal: %d,\n val %d,\n position_for_zero %d,\n position_for_one %d,\n position: %d \n--------\n",
//       local_id, group_id, total_zeroes, zeroes_before, ones_before, pref_sum_local_zeroes[global_id], pref_sum_local_ones[global_id], as[global_id], val, position_for_zero, position_for_one, position
//       );
//    }
//    if (global_id == 34563) {
//        printf("--------\nlocal_id: %d,\n group_id: %d,\n total_zeros %d,\n zeroes_before %d,\n ones_before %d,\n zeroes_local: %d,\n ones_local: %d,\n valReal: %d,\n val %d,\n position_for_zero %d,\n position_for_one %d,\n position: %d \n--------\n",
//               local_id, group_id, total_zeroes, zeroes_before, ones_before, pref_sum_local_zeroes[global_id], pref_sum_local_ones[global_id], as[global_id], val, position_for_zero, position_for_one, position
//               );
//    }
    res[position] = as[global_id];
}


__kernel void pref_sum(__global const unsigned int* as,
                       __global unsigned int* ones_sum,
                       __global unsigned int* zeroes_sum,
                       __global unsigned int* ones_sum_roots,
                       __global unsigned int* zeroes_sum_roots,
                       unsigned int shift) {

    //  ---------------------------- построить дерево ---------------------------------

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int group_num = get_num_groups(0);

    __local unsigned int tree_zeroes[GROUP_SIZE + GROUP_SIZE / 2]; // 2n-1 элементво в полном дереве
    __local unsigned int tree_ones[GROUP_SIZE + GROUP_SIZE / 2];

    unsigned int pos = local_id + 1;
    int levels = GROUP_SIZE; //

    unsigned int read_shift = GROUP_SIZE;
    unsigned int write_shift = 0;
    unsigned int tmp = 0;

    unsigned int zeroes = 0;
    unsigned int ones = 0;

    while (levels) {
        // подготовить сумму
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

        if (pos & 1) {
            zeroes += tree_zeroes[read_shift + pos - 1];
            ones += tree_ones[read_shift + pos - 1];
        }

        pos >>= 1;
        levels >>= 1;
    }

    zeroes_sum[global_id] = zeroes;
    ones_sum[global_id] = ones;
    if (local_id == 0) {
        zeroes_sum_roots[group_id] = tree_zeroes[read_shift];
        ones_sum_roots[group_id] = tree_ones[read_shift];
    }
}

// почти предыдущая функция
__kernel void build_trees_local_step(
                          __global unsigned int* in_zeroes,
                          __global unsigned int* in_ones,
                          __global unsigned int* zeroes_roots,
                          __global unsigned int* ones_roots) {

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);

    __local unsigned int tree_zeroes[GROUP_SIZE + GROUP_SIZE / 2]; // 2n-1 элементво в полном дереве
    __local unsigned int tree_ones[GROUP_SIZE + GROUP_SIZE / 2];

    unsigned int pos = local_id + 1;
    int levels = GROUP_SIZE; //

    unsigned int read_shift = GROUP_SIZE;
    unsigned int write_shift = 0;
    unsigned int tmp = 0;

    unsigned int zeroes = 0;
    unsigned int ones = 0;

    while (levels) {
        // подготовить сумму
        if (levels == GROUP_SIZE) {
            tree_zeroes[local_id] = in_zeroes[global_id];
            tree_ones[local_id] = in_ones[global_id];
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

    in_zeroes[global_id] = zeroes;
    in_ones[global_id] = ones;

    if (local_id == 0) {
        zeroes_roots[group_id] = tree_zeroes[read_shift];
        ones_roots[group_id] = tree_ones[read_shift];
    }
}


// ещё одна очень похожая функция
__kernel void build_trees(__global unsigned int* in_zeroes,
                          __global unsigned int* in_ones,
                          __global unsigned int* trees_zeroes_global,
                          __global unsigned int* trees_ones_global,
                          unsigned int n) {

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_num = get_num_groups(0);
    unsigned int tree_size = (2 * GROUP_SIZE) - 1;
    unsigned int global_shift = tree_size * group_id;
    unsigned int levels = GROUP_SIZE;
    // нулевой уровень

    __local unsigned int trees_zeroes[ (2 * GROUP_SIZE) - 1];
    __local unsigned int trees_ones[ (2 * GROUP_SIZE) - 1];

    trees_zeroes[global_shift + local_id] = in_zeroes[global_id] * (n != -1 && local_id < n);
    trees_ones[global_shift + local_id] = in_ones[global_id] * (n != -1 && local_id < n);

    trees_zeroes_global[global_shift + local_id] = trees_zeroes[global_shift + local_id];
    trees_ones_global[global_shift + local_id] = trees_ones[global_shift + local_id];

//    if (local_id == 0 && group_id == 0 && n == 16) {
//        for (int i = 0; i < n; ++i) {
//            printf("%d, ", trees_ones[i]);
//        }
//        printf("\n");
//    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (n != -1 && local_id >= n) {
////        printf("local_id: %d \n", local_id);
//        trees_zeroes[global_shift + local_id] = 0;
//        trees_ones[global_shift + local_id] = 0;
//    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (local_id == 0 && group_id == 0 && n == 16) {
//        for (int i = 0; i < (2 * GROUP_SIZE - 1); ++i) {
//            printf("%d, ", trees_ones[i]);
//        }
//        printf("----------------\n");
//    }
//
//    if (local_id == 0 && group_id == 0 && n == 16) {
//        for (int i = 0; i < n; ++i) {
//            printf("%d, ", trees_zeroes[i]);
//        }
//        printf("\n");
//    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (local_id == 0 && group_id == 0 && n == 16) {
//        for (int i = 0; i < GROUP_SIZE; ++i) {
//            printf("%d, ", trees_zeroes[i]);
//        }
//        printf("\n");
//    }
    unsigned int shift_write = GROUP_SIZE;
    unsigned int shift_read = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
    while (levels >>= 1) {

//        if (local_id == 0) {
//            printf("global_shift: %d, shift_write: %d; shift_read %d\n, levels: %d\n", global_id, shift_write, shift_read, levels);
//        }
        if (local_id < levels) {
            trees_zeroes[global_shift + shift_write + local_id] =
                    (trees_zeroes[shift_read + 2 * local_id] +
                     trees_zeroes[shift_read + 2 * local_id + 1]);

            trees_ones[global_shift + shift_write + local_id] =
                    (trees_ones[shift_read + 2 * local_id] +
                     trees_ones[shift_read + 2 * local_id + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (local_id < levels) {
            trees_zeroes_global[global_shift + shift_write + local_id] = trees_zeroes[shift_write + local_id];
            trees_ones_global[global_shift + shift_write + local_id] = trees_ones[shift_write + local_id];
        }

        shift_read = shift_write;
        shift_write += levels;

        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);
//
//        if (local_id == 0 && group_id == 0 && n == 16) {
//            for (int i = 0; i < (2 * GROUP_SIZE) - 1; ++i) {
//                printf("%d, ", trees_ones[i]);
//            }
//            printf("----------------\n");
//        }
//        if (local_id == 0 && group_id == 0 && n == 16) {
//            for (int i = 0; i < (2 * GROUP_SIZE) - 1; ++i) {
//                printf("%d, ", trees_zeroes[i]);
//            }
//            printf("--------------\n");
//        }
    }
//    barrier(CLK_GLOBAL_MEM_FENCE);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (local_id == 0 && group_id == 0 && n == 16) {
//        for (int i = 0; i < (2 * GROUP_SIZE) - 1; ++i) {
//            printf("%d, ", trees_ones_global[i]);
//        }
//        printf("----------------\n");
//    }
//    if (local_id == 0 && group_id == 0 && n == 16) {
//        for (int i = 0; i < (2 * GROUP_SIZE) - 1; ++i) {
//            printf("%d, ", trees_zeroes_global[i]);
//        }
//        printf("--------------\n");
//    }
    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        in_zeroes[group_id] = trees_zeroes[global_shift + tree_size - 1];
        in_ones[group_id] = trees_ones[global_shift + tree_size - 1];
    }
}

__kernel void update_from_trees(__global unsigned int* in_zeroes,
                          __global unsigned int* in_ones,
                          __global const unsigned int* trees_zeroes,
                          __global const unsigned int* trees_ones,
                          unsigned int leaf_size
                          ) {

    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int levels = GROUP_SIZE;
    if (group_id == 0) {
        return;
    }
    unsigned int pos = ((global_id / leaf_size) % GROUP_SIZE) ; // индекс группы, целое от деления, лист, с которого начинаем
    unsigned int my_tree = global_id / leaf_size / GROUP_SIZE; // GROUP_SIZE это число листьев в дереве |
    unsigned int tree_size = 2 * GROUP_SIZE - 1; // все деревья такие
//    if (group_id == 2) {
//        printf("global_id %d, local_id %d, pos %d, my_tree %d, leaf_size %d\n", global_id, local_id, pos, my_tree, leaf_size);
//    }
    unsigned int acc_zeroes = 0;
    unsigned int acc_ones = 0;
    unsigned int pos_before = pos;
    // можно if local_id == 0, а можно и без этого.
//    if (global_id == 16777219 && leaf_size == 16777216) {
//        printf("pos1: %d, leaf_size %d, my_tree %d\n", pos, leaf_size, my_tree);
//    }
    int shift = GROUP_SIZE;
    int curr_shift = 0;
    while (levels && pos) {
        if (pos & 1) {
            acc_zeroes += trees_zeroes[my_tree * tree_size + curr_shift + pos - 1];
            acc_ones += trees_ones[my_tree * tree_size + curr_shift + pos - 1];
//            if (global_id == 514) {
//                printf("pos: %d, acc_ones in for: %d\n array: \n", pos, acc_ones);
//                for (int i = 0; i < 10; ++i) {
//                    printf("%d, ", trees_ones[i]);
//                }
//                printf("\n");
//            }
        }
        curr_shift += shift;
        shift /= 2;
        levels >>= 1;
        pos >>= 1;
    }
//    barrier(CLK_GLOBAL_MEM_FENCE);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (acc_ones == 0 && local_id == 50) {
//        printf("hERE: %d, %d\n", acc_zeroes, acc_ones);
//        if (acc_ones == 0) {
//            printf("pos: %d, my_tree: %d\n", pos_before, my_tree);
//        }
//    }
//    barrier(CLK_GLOBAL_MEM_FENCE);
//    barrier(CLK_LOCAL_MEM_FENCE);
    in_zeroes[global_id] += acc_zeroes;
    in_ones[global_id] += acc_ones;
}