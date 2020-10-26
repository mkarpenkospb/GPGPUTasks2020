#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

#define GROUP_SIZE 128

void prepare_local_prefixes(gpu::gpu_mem_32u& as_gpu,
                            gpu::gpu_mem_32u& zeroes_sum_gpu,
                            gpu::gpu_mem_32u& ones_sum_gpu,
                            gpu::gpu_mem_32u& zeroes_sum_roots_gpu,
                            gpu::gpu_mem_32u& ones_sum_roots_gpu,
                            unsigned int shift,
                            unsigned int n) {

    unsigned int workGroupSize = GROUP_SIZE;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    ocl::Kernel pref_sum(radix_kernel, radix_kernel_length, "pref_sum");
    pref_sum.compile();
//    std::vector<unsigned int> s1(n, 0);
//    std::vector<unsigned int> s2(n, 0);
//    std::vector<unsigned int> s3(n / workGroupSize, 0);
//    std::vector<unsigned int> s4(n / workGroupSize, 0);
    pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                  as_gpu, ones_sum_gpu, zeroes_sum_gpu, ones_sum_roots_gpu, zeroes_sum_roots_gpu, shift);
//    ones_sum_gpu.readN(s1.data(), n);
//    zeroes_sum_gpu.readN(s2.data(), n);
//    ones_sum_roots_gpu.readN(s3.data(), n / workGroupSize);
//    zeroes_sum_roots_gpu.readN(s4.data(), n / workGroupSize);
//    std::cout<<"here" << std::endl;
}

void count_prefixes(gpu::gpu_mem_32u& zeroes_gpu,
                    gpu::gpu_mem_32u& ones_gpu,
                    gpu::gpu_mem_32u& zeroes_roots_gpu,
                    gpu::gpu_mem_32u& ones_roots_gpu,
                    gpu::gpu_mem_32u& trees_zeroes,
                    gpu::gpu_mem_32u& trees_ones,
                    unsigned int n
                    ) {

    unsigned int workGroupSize = GROUP_SIZE;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    ocl::Kernel local_trees(radix_kernel, radix_kernel_length, "build_trees_local_step");
    local_trees.compile();

    ocl::Kernel build_trees(radix_kernel, radix_kernel_length, "build_trees");
    build_trees.compile();

    ocl::Kernel update(radix_kernel, radix_kernel_length, "update_from_trees");
    update.compile();

    local_trees.exec(gpu::WorkSize(workGroupSize, global_work_size),
                     zeroes_gpu, ones_gpu, zeroes_roots_gpu, ones_roots_gpu);
    unsigned int start_roots_size = n / workGroupSize;
    unsigned int levels = ceil(log(start_roots_size * 1.0) / log(workGroupSize));
    double trees = start_roots_size * 1.0 / workGroupSize;
    unsigned int leaf_size = workGroupSize;
    unsigned int last_roots = start_roots_size % workGroupSize;
//
//    std::vector<unsigned int> s1(n, 0);
//    std::vector<unsigned int> s2(n, 0);
//    std::vector<unsigned int> s3(n / workGroupSize, 0);
//    std::vector<unsigned int> s4(n / workGroupSize, 0);

    while (levels) {
        if (trees < 1) {
            build_trees.exec(gpu::WorkSize(workGroupSize, workGroupSize),
                             zeroes_roots_gpu, ones_roots_gpu, trees_zeroes, trees_ones, last_roots);
            update.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        zeroes_gpu, ones_gpu, trees_zeroes, trees_ones, leaf_size);
            break;
        } else {
            build_trees.exec(gpu::WorkSize(workGroupSize, ((unsigned int)ceil(trees)) * workGroupSize),
                             zeroes_roots_gpu, ones_roots_gpu, trees_zeroes, trees_ones, -1);
        }
        update.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    zeroes_gpu, ones_gpu, trees_zeroes, trees_ones, leaf_size);

//        std::vector<unsigned int> tree_ones1(workGroupSize * 2 - 1, 0);
//        std::vector<unsigned int> tree_zeroes1(workGroupSize * 2 - 1, 0);
//        trees_ones.readN(tree_ones1.data(), tree_ones1.size());
//        trees_zeroes.readN(tree_zeroes1.data(), tree_zeroes1.size());
//        std::cout <<  "HOHOHO\n\n\n ";
//        for(auto elem: tree_ones1) {
//            std::cout << elem << " ";
//        }
//        std::cout << std::endl;
        last_roots = ((unsigned int) ceil(trees)) % workGroupSize;
        trees = trees / workGroupSize;
        leaf_size *= workGroupSize;
        levels--;
    }
//
//    zeroes_gpu.readN(s1.data(), n);
//    ones_gpu.readN(s2.data(), n);
//    for (int i = 0; i < n; ++i) {
//        if (s1[i] + s2[i] != 256 * (i + 1)) {
//            std::cout << "oops" << std::endl;
//            for (int k = i - 10; k < i + 10; ++k) {
//                std::cout << "{"<<s1[k]  << ", "<< s2[k] << "}, " << std::endl;
//            }
//
//            std::vector<unsigned int> tree_ones1(workGroupSize * 2 - 1, 0);
//            std::vector<unsigned int> tree_zeroes1(workGroupSize * 2 - 1, 0);
//            trees_ones.readN(tree_ones1.data(), tree_ones1.size());
//            trees_zeroes.readN(tree_zeroes1.data(), tree_zeroes1.size());
//
//            for(auto elem: tree_ones1) {
//                std::cout << elem << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    zeroes_roots_gpu.readN(s3.data(), n / workGroupSize);
//    ones_roots_gpu.readN(s4.data(), n / workGroupSize);
//    std::cout<< s1[n / 4 - 1] << std::endl;
}


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u res_gpu;
    gpu::gpu_mem_32u ones_sum_gpu;
    gpu::gpu_mem_32u zeroes_sum_gpu;
    gpu::gpu_mem_32u zeroes_sum_roots_gpu;
    gpu::gpu_mem_32u ones_sum_roots_gpu;
    gpu::gpu_mem_32u ones_roots_gpu;
    gpu::gpu_mem_32u zeroes_roots_gpu;
    gpu::gpu_mem_32u trees_zeroes;
    gpu::gpu_mem_32u trees_ones;

    {
        unsigned int workGroupSize = GROUP_SIZE;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        unsigned int one_tree_size = 2 * workGroupSize - 1;
        unsigned int start_roots_size = n / workGroupSize / workGroupSize;
        unsigned int trees_max_size = ceil(start_roots_size * 1.0/ workGroupSize) * one_tree_size;
        as_gpu.resizeN(n);
        res_gpu.resizeN(n);
        ones_sum_gpu.resizeN(n);
        zeroes_sum_gpu.resizeN(n);
        zeroes_sum_roots_gpu.resizeN(n / workGroupSize);
        ones_sum_roots_gpu.resizeN(n / workGroupSize);
        ones_roots_gpu.resizeN(n / workGroupSize / workGroupSize);
        zeroes_roots_gpu.resizeN(n / workGroupSize / workGroupSize);
        trees_zeroes.resizeN(trees_max_size);
        trees_ones.resizeN(trees_max_size);

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        timer t;
        unsigned int levels = 32;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();
            // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            for (int shift = 0; shift < levels; ++ shift) {
                prepare_local_prefixes(as_gpu, zeroes_sum_gpu, ones_sum_gpu, zeroes_sum_roots_gpu, ones_sum_roots_gpu,
                                       shift, n);
                count_prefixes(zeroes_sum_roots_gpu, ones_sum_roots_gpu, zeroes_roots_gpu, ones_roots_gpu,
                               trees_zeroes, trees_ones,
                               n / workGroupSize);
                radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                           as_gpu, res_gpu, zeroes_sum_gpu, ones_sum_gpu, zeroes_sum_roots_gpu, ones_sum_roots_gpu, shift, n);
                as_gpu.swap(res_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
