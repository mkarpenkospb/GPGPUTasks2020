//#ifdef __CLION_IDE__
//#include <math.h>
//#include "./clion_defines.cl"
//#endif

#line 6
#define THRESHOLD 256.0
#define THRESHOLD2 (256.0 * 256.0)

__kernel void mandelbrot(__global float* gpu_result, unsigned int width, unsigned int height,
                         float fromX, float  fromY, float sizeX, float sizeY, unsigned int maxIter,
                         unsigned int smoothing
                         )
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= width || j >= height) {
        return;
    }

    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;

    for (; iter < maxIter; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > THRESHOLD2) {
            break;
        }
    }

    float result = iter;

    if (smoothing && iter != maxIter) {
        result = result - log(log(sqrt(x * x + y * y)) / log(THRESHOLD)) / log(2.0f);
    }

    result = 1.0f * result / maxIter;
    gpu_result[j * width + i] = result;
    // TODO если хочется избавиться от зернистости и дрожжания при интерактивном погружении - добавьте anti-aliasing:
    // грубо говоря при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
