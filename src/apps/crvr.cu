#include <iostream>
#include <memory>
#include <stdio.h>
#include <util.cuh>

#include <LIEF/ELF.hpp>
#include <LIEF/logging.hpp>

using namespace LIEF::ELF;
const char* r0path = "./ta/level0/t1";

// extern __global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep);
int main(int argc, char** argv ){
    LIEF::logging::set_level(LIEF::logging::LOGGING_LEVEL::LOG_DEBUG);
    std::unique_ptr<const Binary> binary = std::unique_ptr<const Binary>{Parser::parse(r0path)};
    if(binary == nullptr){
        return EXIT_FAILURE;
    }
    uint8_t* mem;
    ccE(cudaMalloc(&mem, 1024*10));
    // Load up the binary
    for (const Segment& segment : binary->segments()) {
        if(segment.type() == SEGMENT_TYPES::PT_LOAD){
            uint64_t addr = segment.virtual_address();
            uint64_t vsize = segment.virtual_size();
            uint64_t contentsize = segment.get_content_size();
            printf("contentsize: %lu, vsize: %lu\n", contentsize, vsize);
            ccE(cudaMemcpy(mem + addr, segment.content().data(), contentsize, cudaMemcpyHostToDevice));
        }
    }
    REG* regfile;
    REG* pcfile;
    ccE(cudaMalloc(&regfile, NUM_OF_REGS*sizeof(REG)*32));
    ccE(cudaMalloc(&pcfile, sizeof(REG)*32));
    // initialize pc?
    uint32_t entrypt = (uint32_t)binary->entrypoint();
    initPC<<<1,32>>>(pcfile, entrypt);
    for(int i = 0; i < 6; i++){
        step<<<1,32>>>(regfile, pcfile, mem, nullptr, 1);
    }
    ccE(cudaDeviceSynchronize());
    // __global__ void step(REG* regfile, REG* pcfile, uint8_t* mem)

}