#include <util.cuh>

__global__ void dumpS(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, int coreid);