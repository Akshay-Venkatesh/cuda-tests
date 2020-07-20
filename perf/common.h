#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CU_DEVICE_NAME_LEN 256

#define CU_CHECK(fxn)                                                         \
    do {                                                                      \
        CUresult curesult = fxn;                                              \
        char *cu_err_string;                                                  \
        if (CUDA_SUCCESS != curesult) {                                       \
            if (CUDA_SUCCESS == cuGetErrorString(curesult, (const char**) &cu_err_string)) { \
                fprintf(stderr, "%s\n", cu_err_string);                       \
            } else {                                                          \
                fprintf(stderr, "invalid cuda error\n");                      \
            }                                                                 \
            exit(-1);                                                         \
        }                                                                     \
    } while(0);

typedef struct cu_dev_ctx {
    int num_devices;
    int ordinal;
    char device_name[CU_DEVICE_NAME_LEN];
    CUdevice device;
    CUuuid uuid;
    size_t bytes;
    CUcontext ctx;
} cu_dev_ctx_t;


int cu_ctx_init(cu_dev_ctx_t *cu_ctx);
int cu_ctx_cleanup(cu_dev_ctx_t *cu_ctx);
