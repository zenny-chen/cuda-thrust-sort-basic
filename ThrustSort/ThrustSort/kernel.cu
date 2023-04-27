#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cstdio>

struct MyItem
{
    int a;
    int b;

    __host__ __device__ bool operator < (const MyItem& rhs) const
    {
        return (a + b) < (rhs.a + rhs.b);
    }

    __host__ void PrintItemValue(void)
    {
        printf("(%d + %d = %d)\n", a, b, a + b);
    }
};

static void ItemSortTest(void)
{
    constexpr int elemCount = 16;

    MyItem items[elemCount] = {
        MyItem{ 0, 1 }, MyItem{ 5, 6 }, MyItem{ 2, 3 }, MyItem{ 9, 9 },
        MyItem{ 8, 8 }, MyItem{ 7, 7 }, MyItem{ 6, 7 }, MyItem{ 6, 6 },
        MyItem{ 5, 5 }, MyItem{ 4, 4 }, MyItem{ 3, 3 }, MyItem{ 4, 5 },
        MyItem{ 2, 2 }, MyItem{ 0, 0 }, MyItem{ 1, 2 }, MyItem{ 3, 6 }
    };

    MyItem* devMem = nullptr;
    constexpr auto bufferSize = elemCount * sizeof(*devMem);

    do
    {
        auto cudaStatus = cudaMalloc(&devMem, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devMem, items, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        try
        {
            thrust::device_ptr<MyItem> sortItems(devMem);
            thrust::sort(sortItems, sortItems + elemCount, thrust::less<MyItem>());
        }
        catch (const thrust::system::system_error& sysErr)
        {
            printf("thrust system error: %s\n", sysErr.what());
            break;
        }
        catch (const thrust::system::detail::bad_alloc& allocErr)
        {
            printf("thrust bad alloc: %s\n", allocErr.what());
            break;
        }

        cudaStatus = cudaMemcpy(items, devMem, bufferSize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        for (int i = 0; i < elemCount; i++) {
            items[i].PrintItemValue();
        }

    } while (false);

    if (devMem != nullptr) {
        cudaFree(devMem);
    }
}

static void KeySortTest(void)
{
    //  0123456789abcdef
    // "Hi, CUDA Thrust!"
    int chars[] = {
        'A', 'C', 'D', 'H', 'h', 'i', 'r', ' ',
        's', 'T', 't', 'U', 'u', '!', ',', ' '
    };
    const int keys[] = {
        7, 4, 6, 0, 10, 1, 11, 3,
        13, 9, 14, 5, 12, 15, 2, 8
    };

    constexpr auto bufferSize = sizeof(keys);
    constexpr int elemCount = int(bufferSize / sizeof(keys[0]));

    int* devChars = nullptr;
    int* devKeys = nullptr;

    do
    {
        auto cudaStatus = cudaMalloc(&devChars, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&devKeys, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devChars, chars, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devKeys, keys, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        try
        {
            thrust::device_ptr<int> sortItems(devChars);
            thrust::device_ptr<int> sortKeys(devKeys);
            thrust::sort_by_key(sortKeys, sortKeys + elemCount, sortItems, thrust::less<int>());
        }
        catch (const thrust::system::system_error& sysErr)
        {
            printf("thrust system error: %s\n", sysErr.what());
            break;
        }
        catch (const thrust::system::detail::bad_alloc& allocErr)
        {
            printf("thrust bad alloc: %s\n", allocErr.what());
            break;
        }

        cudaStatus = cudaMemcpy(chars, devChars, bufferSize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        printf("The string is: ");
        for (int i = 0; i < elemCount; i++) {
            printf("%c", chars[i]);
        }
        puts("");

    } while (false);

    if (devChars != nullptr) {
        cudaFree(devChars);
    }
    if (devKeys != nullptr) {
        cudaFree(devKeys);
    }
}

int main(void)
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    auto cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        puts("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 0;
    }

    ItemSortTest();

    puts("\n================\n");

    KeySortTest();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        puts("cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

