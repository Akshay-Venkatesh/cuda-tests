#include "common.h"
#include <sys/time.h>
#include <unistd.h>

#define MALLOC_SIZE (1 << 22)
#define ALLOC_SIZE sizeof(char) * MALLOC_SIZE 
#define MAX_ITERS 1024

int ptr_is_managed(void *ptr, char *title)
{
    CUresult cu_res;
    uint32_t is_managed;

    cu_res = cuPointerGetAttribute(&is_managed,
        CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr) ptr);
    if (is_managed) {
        printf("%s: %p is treated as managed memory\n", title, ptr);
    } else {
        printf("%s: %p is not treated as managed memory\n", title, ptr);
    }
    return is_managed;
}

void print_ptr_type(void *ptr, char *title)
{
    CUresult cu_res;
    int mem_type;

    cu_res = cuPointerGetAttribute(&mem_type,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr) ptr);
    if (CUDA_ERROR_INVALID_VALUE == cu_res) {
        printf("%s: %p is treated as memory not allocated by cuda\n", title, ptr);
    } else if (CUDA_SUCCESS == cu_res) {
        if (CU_MEMORYTYPE_DEVICE == mem_type) {
            printf("%s: %p is treated as cuda pinned memory\n", title, ptr);
            ptr_is_managed(ptr, title);
        } else if (CU_MEMORYTYPE_HOST == mem_type) {
            printf("%s: %p is treated as cuda host memory\n", title, ptr);
            ptr_is_managed(ptr, title);
        } else {
            printf("%s: %p shouldn't be here \n", title, ptr);
        }
    }
    return;
}

int get_ptr_attrib(void *ptr)
{
    CUresult cu_res;
    int mem_type;

    cu_res = cuPointerGetAttribute(&mem_type,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr) ptr);
    return mem_type;
}

typedef struct mem_attrib {
    int mem_type;
    uint32_t managed;
} mem_attrib_t;

int get_ptr_attribs(void *ptr, mem_attrib_t *mem_attr)
{
    CUresult cu_res;
    CUpointer_attribute attributes[2] =
        {CU_POINTER_ATTRIBUTE_IS_MANAGED, CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
    void *data[] = {(void *) &mem_attr->mem_type, (void *) &mem_attr->managed};

    cuPointerGetAttributes(2, attributes, data, (CUdeviceptr) ptr);

    return 0;
}

void print_ptr_type2(void *ptr, char *title)
{
    CUresult cu_res;
    CUpointer_attribute attributes[2] =
        {CU_POINTER_ATTRIBUTE_IS_MANAGED, CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
    int mem_type;
    uint32_t managed;
    void *data[] = {(void *) &mem_type, (void *) &managed};

    //printf("default mem_type = %d managed = %d\n", mem_type, managed);
    //printf("device_type = %d host_type = %d\n", CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST);
    cu_res = cuPointerGetAttributes(2, attributes, data, (CUdeviceptr) ptr);
    printf("%s: %p %d %d\n", title, ptr, mem_type, managed);
    if (!mem_type && !managed) {
        printf("%s: %p is not allocated by cuda. Likely host memory\n", title, ptr);
        return;
    } else if (!mem_type && managed) {
        printf("%s: %p Likely cuda managed memory\n", title, ptr);
    } else {
        if (CU_MEMORYTYPE_DEVICE == mem_type) printf("%s: %p is pinned memory\n", title, ptr);
        else if (CU_MEMORYTYPE_HOST == mem_type) printf("%s :%p is cuda allocated host memory\n", title, ptr);
        else printf("%s is an unknown memory type\n", title);
    }

    return;
}

void ptr_attrib_check_perf(void *ptr, char *title)
{
    int i;
    int mem_type;
    struct timeval t_start, t_stop;
    double latency;

    printf("checking perf for %s\n", title);
    gettimeofday(&t_start, NULL);
    for (i = 0; i < MAX_ITERS; i++) {
        mem_type = get_ptr_attrib(ptr);
    }
    gettimeofday(&t_stop, NULL);
    latency  = t_stop.tv_usec - t_start.tv_usec;
    latency += ((t_stop.tv_sec - t_start.tv_sec) * 1E+6);
    latency /= MAX_ITERS;
    printf("average checking time for %s = %lf \n\n", title, latency);
}

void ptr_attribs_check_perf(void *ptr, char *title)
{
    int i;
    mem_attrib_t mem_attr;
    struct timeval t_start, t_stop;
    double latency;

    printf("checking perf for %s\n", title);
    gettimeofday(&t_start, NULL);
    for (i = 0; i < MAX_ITERS; i++) {
        get_ptr_attribs(ptr, &mem_attr);
    }
    gettimeofday(&t_stop, NULL);
    latency  = t_stop.tv_usec - t_start.tv_usec;
    latency += ((t_stop.tv_sec - t_start.tv_sec) * 1E+6);
    latency /= MAX_ITERS;
    printf("average checking time for %s = %lf \n\n", title, latency);
}

int main(int argc, char **argv)
{
    cu_dev_ctx_t cu_ctx;
    char *host_mem;
    char *cu_host_mem;
    char *cu_pinned_mem;
    char *cu_managed_mem;

    if (cu_ctx_init(&cu_ctx)) return -1;

    host_mem = malloc(ALLOC_SIZE);
    CU_CHECK(cuMemAllocHost((void **) &cu_host_mem, ALLOC_SIZE));
    CU_CHECK(cuMemAlloc((CUdeviceptr *) &cu_pinned_mem, ALLOC_SIZE));
    CU_CHECK(cuMemAllocManaged((CUdeviceptr *) &cu_managed_mem, ALLOC_SIZE,
                                CU_MEM_ATTACH_GLOBAL));
    printf("\n\n");

    printf("checking 1 pointer attribute at a time ...\n");
    print_ptr_type((void *) host_mem, "host ptr");
    print_ptr_type((void *) cu_host_mem, "cuda host ptr");
    print_ptr_type((void *) cu_managed_mem, "cuda managed ptr");
    print_ptr_type((void *) cu_pinned_mem, "cuda pinned ptr");
    printf("\n\n");

    printf("checking 2 pointer attributes at a time ...\n");
    print_ptr_type2((void *) host_mem, "host ptr");
    print_ptr_type2((void *) cu_host_mem, "cuda host ptr");
    print_ptr_type2((void *) cu_managed_mem, "cuda managed ptr");
    print_ptr_type2((void *) cu_pinned_mem, "cuda pinned ptr");
    printf("\n\n");

    printf("checking what is the managed status of pointers ...\n");
    ptr_is_managed((void *) host_mem, "host ptr");
    ptr_is_managed((void *) cu_host_mem, "cuda host ptr");
    ptr_is_managed((void *) cu_managed_mem, "cuda managed ptr");
    ptr_is_managed((void *) cu_pinned_mem, "cuda pinned ptr");
    printf("\n\n");

    ptr_attrib_check_perf((void *) host_mem, "host ptr");
    ptr_attrib_check_perf((void *) cu_host_mem, "cuda host ptr");
    ptr_attrib_check_perf((void *) cu_managed_mem, "cuda managed ptr");
    ptr_attrib_check_perf((void *) cu_pinned_mem, "cuda pinned ptr");

    ptr_attribs_check_perf((void *) host_mem, "host ptr");
    ptr_attribs_check_perf((void *) cu_host_mem, "cuda host ptr");
    ptr_attribs_check_perf((void *) cu_managed_mem, "cuda managed ptr");
    ptr_attribs_check_perf((void *) cu_pinned_mem, "cuda pinned ptr");

    free(host_mem);
    CU_CHECK(cuMemFreeHost((void *) cu_host_mem));
    CU_CHECK(cuMemFree((CUdeviceptr) cu_pinned_mem));
    CU_CHECK(cuMemFree((CUdeviceptr) cu_managed_mem));
    cu_ctx_cleanup(&cu_ctx);
    return 0;
}

