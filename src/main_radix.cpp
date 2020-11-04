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

#define GROUP_SIZE 256

void prepare_local_prefixes(gpu::gpu_mem_32u& as_gpu,
                            gpu::gpu_mem_32u& zeroes_sum_gpu,
                            gpu::gpu_mem_32u& ones_sum_gpu,
                            unsigned int shift,
                            unsigned int n) {

    unsigned int workGroupSize = GROUP_SIZE;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    ocl::Kernel pref_sum(radix_kernel, radix_kernel_length, "pref_sum");
    pref_sum.compile();

    pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                  as_gpu, ones_sum_gpu, zeroes_sum_gpu, shift);
}

void count_prefixes(gpu::gpu_mem_32u& zeroes_gpu,
                    gpu::gpu_mem_32u& ones_gpu,
                    unsigned int n
                    ) {
    // zeroes_gpu, ones_gpu -- массивы, для которых нужно будет посчитать преф сумму.
    // в них уже есть значения, полученные из предыдущего шага
    // zeroes_roots_gpu, ones_roots_gpu -- пустые буфферы, в них на первом шаге текущей функции
    // положим корни массивов zeroes_gpu, ones_gpu
    // корни -- это вершины дерева отрезков, как если бы мы строили такие деревья для отрезков размера workGroupSize

    unsigned int workGroupSize = GROUP_SIZE;
    unsigned int global_work_size = 0;
    // (n / workGroupSize) -- тут надо префиксы обновлять
    unsigned int global_work_size_for_update = ((n / workGroupSize)  + workGroupSize - 1) / workGroupSize * workGroupSize;

    ocl::Kernel count_pref(radix_kernel, radix_kernel_length, "count_pref_on_roots");
    count_pref.compile();

    ocl::Kernel update_pref(radix_kernel, radix_kernel_length, "update_from_pref");
    update_pref.compile();

    unsigned int step_between_roots = workGroupSize; // начальные корни
    unsigned int roots = n / step_between_roots;
    // нужно знать, сколько останется элементов на последнем шаге
    while (roots) {
        global_work_size = (roots + workGroupSize - 1) / workGroupSize * workGroupSize;
        count_pref.exec(gpu::WorkSize(workGroupSize, global_work_size), zeroes_gpu, ones_gpu, step_between_roots,
                        roots);
        update_pref.exec(gpu::WorkSize(workGroupSize, global_work_size_for_update), zeroes_gpu, ones_gpu, step_between_roots);
        step_between_roots *= workGroupSize;
        roots /= workGroupSize;
    }
}


template<typename T>
void raiseFail(const T &a, const T &b, const std::string& message, const std::string& filename, int line)
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

    int benchmarkingIters = 10;
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

    {
        unsigned int workGroupSize = GROUP_SIZE;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        as_gpu.resizeN(n);
        res_gpu.resizeN(n);

        // массивы с преф суммами единическ и ноликов по группам в workGroupSize
        // все корни можно хранить прямо в этом массиве и его обновлять
        ones_sum_gpu.resizeN(n);
        zeroes_sum_gpu.resizeN(n);

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        timer t;
        unsigned int levels = 32;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();
            // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            for (int shift = 0; shift < levels; ++ shift) {
                prepare_local_prefixes(as_gpu, zeroes_sum_gpu, ones_sum_gpu, shift, n);
                count_prefixes(zeroes_sum_gpu, ones_sum_gpu, n);
                radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                           as_gpu, res_gpu, zeroes_sum_gpu, ones_sum_gpu, shift, n);
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
