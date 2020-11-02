#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <cassert>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


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

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
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

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel bitonic_begin(bitonic_kernel, bitonic_kernel_length, "local_bitonic_begin");
        bitonic_begin.compile();

        ocl::Kernel bitonic_global_step(bitonic_kernel, bitonic_kernel_length, "bitonic_global_step");
        bitonic_global_step.compile();

        ocl::Kernel bitonic_endings(bitonic_kernel, bitonic_kernel_length, "bitonic_local_endings");
        bitonic_begin.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            //  workitem-ов будет в 2 раза меньше, чем элементов
            unsigned int global_work_size = ((n / 2) + workGroupSize - 1) / workGroupSize * workGroupSize;

            bitonic_begin.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu);
            unsigned int outer = n / (workGroupSize * 2); // счетчик для внешего цикла
            unsigned int segment_length = workGroupSize * 2 * 2;
            while (outer != 1) {
                bitonic_global_step.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, segment_length, 1);
                for (unsigned int i = segment_length / 2; i > workGroupSize * 2;  i >>= 1) {
                    bitonic_global_step.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, i, 0);
                }
                bitonic_endings.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu);
                outer >>= 1;
                segment_length <<= 1;
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }
    // проверка что результат сортирован без cpu
//    for (int i = 0; i < n - 1; ++i) {
//        assert(as[i] <= as[i + 1]);
//    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
