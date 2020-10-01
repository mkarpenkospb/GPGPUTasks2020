#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"


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
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        unsigned int workGroupSize = 32 * 8;
        gpu::gpu_mem_32f gpu_data_buffer;
        gpu::gpu_mem_32f gpu_result_buffer;
        gpu_data_buffer.resize(n * sizeof(unsigned int)); // если инициализация не мусором, можно инициализировать кратно ворк группе....
        gpu_data_buffer.write(as.data(), n * sizeof(unsigned int));
        gpu_result_buffer.resize(1 * sizeof(unsigned int));

        ocl::Kernel sum_tree(sum_kernel, sum_kernel_length, "sum_tree");
        sum_tree.compile();


        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0; // убирала запись и чтение из этого цикла, разницы нет почти
            gpu_result_buffer.write(&sum, 1 * sizeof(unsigned int));
            sum_tree.exec(gpu::WorkSize(workGroupSize, n),
                          gpu_data_buffer,
                          n,
                          gpu_result_buffer);
            gpu_result_buffer.read(&sum, 1 * sizeof(unsigned int));
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;


        // ---------------------- less atomic -----------------------------------

        ocl::Kernel sum_tree_less_atomic(sum_kernel, sum_kernel_length, "sum_tree_less_atomic");

        unsigned int data_per_item = 8;
        sum_tree_less_atomic.compile();
        t.restart();
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            gpu_result_buffer.write(&sum, 1 * sizeof(unsigned int));
            sum_tree_less_atomic.exec(gpu::WorkSize(workGroupSize, n / data_per_item),
                          gpu_data_buffer,
                          n,
                          gpu_result_buffer);
            gpu_result_buffer.read(&sum, 1 * sizeof(unsigned int));
            EXPECT_THE_SAME(reference_sum, sum, "GPU2 result should be consistent!");
            t.nextLap();
        }

        std::cout << "GPU2: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU2: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

    }
}