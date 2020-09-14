//#ifdef __CLION_IDE__
// Этот include виден только для CLion парсера, это позволяет IDE "знать" ключевые слова вроде __kernel, __global
// а так же уметь подсказывать OpenCL методы описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
//#include "clion_defines.cl"
//#endif

#line 8

__kernel void aplusb(__global const float* as, __global const float* bs, __global float* cs, const unsigned int size)
{
    int workItemId = get_global_id(0);
    if (workItemId < size)
        cs[workItemId] = as[workItemId] + bs[workItemId];
}
