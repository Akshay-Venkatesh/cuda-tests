#include "common.h"
  
int cu_ctx_init(cu_dev_ctx_t *cu_ctx)
{
    cu_ctx->ordinal = 0;
    CU_CHECK(cuInit(0));

    CU_CHECK(cuDeviceGetCount(&(cu_ctx->num_devices)));
    if (cu_ctx->num_devices <= 0) {
        fprintf(stderr, "No cuda capable devices\n");
        return -1;
    }

    CU_CHECK(cuDeviceGet(&(cu_ctx->device), cu_ctx->ordinal));
    CU_CHECK(cuDeviceGetName(cu_ctx->device_name, CU_DEVICE_NAME_LEN, cu_ctx->device));
    CU_CHECK(cuDeviceTotalMem(&(cu_ctx->bytes), cu_ctx->device));
    CU_CHECK(cuDeviceGetUuid(&(cu_ctx->uuid), cu_ctx->device));
    CU_CHECK(cuCtxCreate(&(cu_ctx->ctx), CU_CTX_SCHED_SPIN, cu_ctx->device));

    fprintf(stderr, "Created context on %s with %ld bytes of mem\n",
            cu_ctx->device_name, (long int) cu_ctx->bytes);
    return 0;
}

int cu_ctx_cleanup(cu_dev_ctx_t *cu_ctx)
{
    CU_CHECK(cuCtxDestroy(cu_ctx->ctx));
    return 0;
}
