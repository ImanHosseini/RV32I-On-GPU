#include <iostream>
#include <memory>
#include <stdio.h>
#include <util.cuh>
#include <argh.h>
#include <thread>
#include <vector>
#include <tuple>

#include <LIEF/ELF.hpp>
#include <LIEF/logging.hpp>

using namespace LIEF::ELF;
const char* r0path = "/home/iman/projs2/CRV2/ta/level0/t1";

__global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec);
extern void initialize(REG* &regfile, REG* &pcfile, core_status_t* &svec, cudaStream_t* streams, int np, int nq);
extern void set_cms(uint32_t*);
extern uint32_t calc_mpc(int np, int nq);
std::vector<std::thread> hths;
cudaStream_t* cstreams;

typedef std::tuple<uint32_t, uint32_t, const uint8_t*> mts_t; 
typedef std::vector<mts_t> mmc_t;

void tX(cudaStream_t s, uint32_t nc, uint32_t mpc, uint32_t pc0, mmc_t mmc)
{
    REG *regfile, *pcfile;
    core_status_t* svec;
    uint8_t* mem;
    ccE(cudaMallocAsync(&regfile, NUM_OF_REGS*sizeof(REG)*nc, s));
    ccE(cudaMallocAsync(&pcfile, sizeof(REG)*nc, s));
    ccE(cudaMallocAsync(&svec, sizeof(core_status_t)*nc, s));
    ccE(cudaMallocAsync(&mem, mpc, s));
    initPC<<< nc/32, 32, 0, s>>>(pcfile, pc0);
    // prepare the memory
    for(mts_t mts : mmc){
        ccE(cudaMemcpyAsync(mem + std::get<0>(mts), std::get<2>(mts), std::get<1>(mts), cudaMemcpyHostToDevice, s));
    }
    for(int i = 0; i < 10; i++){
        step<<<nc/32, 32, 0, s>>>(regfile, pcfile, mem, svec);
    }
    ccE(cudaStreamSynchronize(s));
}

int main(int argc, char* argv[]){
    argh::parser cmdl(argv);
    int np, nq;
    std::string fpath;
    // std::string rpath{r0path};
    cmdl("np", 1024) >> np;
    cmdl("nq", 32) >> nq;
    cmdl("f", std::string(r0path)) >> fpath;
    uint32_t pc0;
    std::unique_ptr<const Binary> binary = std::unique_ptr<const Binary>{Parser::parse(fpath)};
    pc0 = binary->entrypoint();
    mmc_t mmc;
    // Load up the binary
    for (const Segment& segment : binary->segments()) {
        if(segment.type() == SEGMENT_TYPES::PT_LOAD){
            uint64_t addr = segment.virtual_address();
            uint64_t vsize = segment.virtual_size();
            uint64_t contentsize = segment.get_content_size();
            printf("contentsize: %lu, vsize: %lu\n", contentsize, vsize);
            mmc.push_back(mts_t(addr, vsize, segment.content().data()));
        }
    }
    printf("%s\n", banner);
    printf("|np: %d|nq: %d| f: %s|\n",np, nq, fpath.c_str());
    cstreams = new cudaStream_t[nq];
    uint32_t mpc = calc_mpc(np, nq);
    set_cms(&mpc);
    for(int i = 0; i < nq; i++){
        ccE(cudaStreamCreate(cstreams+i));
        hths.emplace_back(tX, cstreams[i], np/nq, mpc, pc0, mmc);
    }
    for(int i = 0; i < nq; i++){
        hths[i].join();
    }
}