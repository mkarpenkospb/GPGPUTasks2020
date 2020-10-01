#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

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
    int max_n = (1 << 24);


    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int workGroupSize = 32 * 4;



    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        std::vector<int> idx(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
            idx[i] = i + 1;
        }
        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL

            gpu::gpu_mem_32f gpu_pref_buffer; // входящий буффер максимальных префиксов
            gpu::gpu_mem_32f gpu_sum_buffer; // входящий буффер сумм на промежутке
            gpu::gpu_mem_32f gpu_idx_buffer; // входящий буффер максимальной позиции

            // аналогичные результирующие буфферы
            gpu::gpu_mem_32f gpu_res_pref_buffer;
            gpu::gpu_mem_32f gpu_res_sum_buffer;
            gpu::gpu_mem_32f gpu_res_idx_buffer;

            gpu_pref_buffer.resize(n * sizeof(int));
            gpu_sum_buffer.resize(n * sizeof(int));
            gpu_idx_buffer.resize(n * sizeof(int));

            gpu_pref_buffer.write(as.data(), n * sizeof(int));
            gpu_pref_buffer.copyTo(gpu_sum_buffer, n * sizeof(int));
            gpu_idx_buffer.write(idx.data(), n * sizeof(int));

            gpu_res_pref_buffer.resize(n * sizeof(int));
            gpu_res_sum_buffer.resize(n * sizeof(int));
            gpu_res_idx_buffer.resize(n * sizeof(int));


            // без запасного буффера ооочень долго
            gpu::gpu_mem_32f gpu_clean_pref_buffer; gpu_clean_pref_buffer.resize(n * sizeof(int));
            gpu_pref_buffer.copyTo(gpu_clean_pref_buffer, n * sizeof(int));
            gpu::gpu_mem_32f gpu_clean_sum_buffer; gpu_clean_sum_buffer.resize(n * sizeof(int));
            gpu_sum_buffer.copyTo(gpu_clean_sum_buffer, n * sizeof(int));
            gpu::gpu_mem_32f gpu_clean_idx_buffer; gpu_clean_idx_buffer.resize(n * sizeof(int));
            gpu_idx_buffer.copyTo(gpu_clean_idx_buffer, n * sizeof(int));

            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_pref_simple");

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int nextSize = n;
                int result = 0;
                int iterMax = (int) ceil(log(n) / log(workGroupSize));

                gpu_clean_pref_buffer.copyTo(gpu_pref_buffer, n * sizeof(int));
                gpu_clean_sum_buffer.copyTo(gpu_sum_buffer, n * sizeof(int));
                gpu_clean_idx_buffer.copyTo(gpu_idx_buffer, n * sizeof(int));

                for (int kernelLaunch = 0; kernelLaunch < iterMax; ++kernelLaunch) {

                    kernel.exec(gpu::WorkSize(workGroupSize, nextSize),
                                gpu_pref_buffer,
                                gpu_sum_buffer,
                                gpu_idx_buffer,
                                nextSize,
                                gpu_res_pref_buffer,
                                gpu_res_sum_buffer,
                                gpu_res_idx_buffer
                                );
                    nextSize = (nextSize + workGroupSize - 1) / workGroupSize;
                    // если просто менять буфферы в аргументах,
                    // та же скорость примерно
                    gpu_res_pref_buffer.swap(gpu_pref_buffer);
                    gpu_res_sum_buffer.swap(gpu_sum_buffer);
                    gpu_res_idx_buffer.swap(gpu_idx_buffer);
                }
                gpu_pref_buffer.read(&max_sum, sizeof(int));
                gpu_sum_buffer.read(&sum, sizeof(int));
                gpu_idx_buffer.read(&result, sizeof(int));

                EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU simple implementation: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU simple implementation:: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
