#include <util.cuh>

__global__ void initPC(REG* pcfile, REG val){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    pcfile[tid] = val;
}

// SP is X2
__global__ void initSP(REG* regfile, uint32_t addr){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    REG *regs = regfile + tid * NUM_OF_REGS;
    regs[2] = addr;
}

const char* banner = "\n--------------------------------------------------------------------------\n           ooooooooo.   oooooo     oooo        .oooo.         .oooo.   \n           `888   `Y88.  `888.     .8'        d8P'`Y8b      .dP''Y88b  \n .oooooooo  888   .d88'   `888.   .8'        888    888           ]8P' \n888' `88b   888ooo88P'     `888. .8'         888    888         .d8P'  \n888   888   888`88b.        `888.8'          888    888       .dP'     \n`88bod8P'   888  `88b.       `888'           `88b  d88' .o. .oP     .o \n`8oooooo.  o888o  o888o       `8'             `Y8bd8P'  Y8P 8888888888 \nd'     YD                                                              \n'Y88888P'\n--------------------------------------------------------------------------\n";

const char* CSTATE_STRINGS[] = {
#define X(name) #name,
    CSTATE_ENUM_VALUES
#undef X
};

// Define the function
const char* CStateToString(CSTATE state) {
    if (state >= CSTATE_COUNT) {
        return "UNKNOWN";
    }
    return CSTATE_STRINGS[state];
}