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
const char* r0path = "./ta/level0/t1";

// extern __global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep);
extern void initialize(REG* &regfile, REG* &pcfile, core_status_t* &svec, cudaStream_t* streams, int np, int nq);
extern void set_cms(int32_t*);
extern int32_t calc_mpc(int np, int nq);
extern void set_cro(uint32_t cro = 0);
std::vector<std::thread> hths;
cudaStream_t* cstreams;

typedef std::tuple<uint32_t, uint32_t, const uint8_t*> mts_t; 
typedef std::vector<mts_t> mmc_t;
bool print_final_status = false;

void tX(cudaStream_t s, uint32_t nc, uint32_t mpc, uint32_t pc0, mmc_t mmc)
{
    REG *regfile, *pcfile;
    core_status_t* svec;
    uint8_t* gmem;
    // int32_t fm = dumpM();
    ccE(cudaMallocAsync(&regfile, NUM_OF_REGS*sizeof(REG)*nc, s));
    ccE(cudaMallocAsync(&pcfile, sizeof(REG)*nc, s));
    // fm = dumpM();
    // printf("DELTA: %d\n", fm - (nc*mpc));
    ccE(cudaMallocAsync(&svec, sizeof(core_status_t)*nc, s));
    // uint32_t fm = dumpM();
    // printf("DELTA: %d\n", free_memory - (np*mpc));
    ccE(cudaMallocAsync(&gmem, mpc * nc, s));
    initPC<<<nc/32, 32, 0, s>>>(pcfile, pc0);
    initSP<<<nc/32, 32, 0, s>>>(regfile, mpc);
    // prepare the memory
    for(mts_t mts : mmc){
        for(int i = 0; i < nc; i++){
            ccE(cudaMemcpyAsync(gmem + i*mpc + std::get<0>(mts), std::get<2>(mts), std::get<1>(mts), cudaMemcpyHostToDevice, s));
        }
    }
    step<<<nc/32, 32, 0, s>>>(regfile, pcfile, gmem, svec, 0);
    // print statuses
    if(print_final_status){
        core_status_t* svec_h = (core_status_t*)malloc(sizeof(core_status_t) * nc);
        ccE(cudaMemcpy(svec_h, svec, sizeof(core_status_t) * nc, cudaMemcpyDeviceToHost));
        for(int i = 0; i < nc; i++){
            printf("[%d]: [%d]\n", i, svec_h[i].state);
        }
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
    if (cmdl["fst"]){
        print_final_status = true;
    }
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
    int32_t mpc = calc_mpc(np, nq);
    set_cms(&mpc);
    for(int i = 0; i < nq; i++){
        ccE(cudaStreamCreate(cstreams+i));
        hths.emplace_back(tX, cstreams[i], np/nq, mpc, pc0, mmc);
    }
    for(int i = 0; i < nq; i++){
        hths[i].join();
    }
}