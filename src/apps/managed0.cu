#include <iostream>
#include <memory>
#include <stdio.h>
#include <util.cuh>
#include <argh.h>
#include <thread>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>

#include <LIEF/ELF.hpp>
#include <LIEF/logging.hpp>
#define CORE_GMEM_SIZE 1024*1024*16

using namespace LIEF::ELF;
const char* r0path = "./ta/bin/hello";
extern void initialize(REG* &regfile, REG* &pcfile, core_status_t* &svec, cudaStream_t* streams, int np, int nq);
extern void set_cms(int32_t*);
extern int32_t calc_mpc(int np, int nq);
extern void set_cro(uint32_t cro = 0);
extern void init_cmem(void*, uint32_t);
std::vector<std::thread> hths;
cudaStream_t* cstreams;
const uint8_t* cuda_constant_seg;

typedef std::tuple<uint32_t, uint32_t, const uint8_t*> mts_t; 
typedef std::vector<mts_t> mmc_t;
bool print_final_status = false;

uint32_t cmem_off = 0;
uint32_t cro_spill = 0;
uint8_t* gmembase = NULL;
int np, nq;
uint8_t* gmem_pool;

void tX(cudaStream_t s, uint32_t nc, uint32_t mpc, uint32_t pc0, mmc_t mmc, int qnum)
{
    REG *regfile, *pcfile;
    core_status_t* svec;
    uint8_t* gmem = gmem_pool + (CORE_GMEM_SIZE * (np / nq) * qnum);
    ccE(cudaMallocAsync(&regfile, NUM_OF_REGS*sizeof(REG)*nc, s));
    ccE(cudaMallocAsync(&pcfile, sizeof(REG)*nc, s));
    ccE(cudaMallocAsync(&svec, sizeof(core_status_t)*nc, s));
    ccE(cudaMallocAsync(&gmem, (uint64_t)mpc * nc, s));
    printf("mmc.size(): %zu\n", mmc.size());
    assert(mmc.size() <= 1);
    mts_t mts = mmc[0];
    uint32_t addr = std::get<0>(mts);
    uint32_t vsize = std::get<1>(mts);
    const uint8_t* gdata = std::get<2>(mts);
    for(int i = 0; i < nc; i++) {   
        // allocate the spill
        if(cro_spill) {
            ccE(cudaMemcpyAsync(gmem + CORE_GMEM_SIZE * i, cuda_constant_seg + CRO_MAX_SIZE, cro_spill, cudaMemcpyHostToDevice, s));
        }
        ccE(cudaMemcpyAsync(gmem + CORE_GMEM_SIZE * i + addr, gdata, vsize, cudaMemcpyHostToDevice, s));
    }
    // execute
    initPC<<<nc/32, 32, 0, s>>>(pcfile, pc0);
    initSP<<<nc/32, 32, 0, s>>>(regfile, CORE_GMEM_SIZE + cmem_off + cro_spill);
    ccE(cudaMemsetAsync(svec, 0x0, sizeof(core_status_t)*nc, s));
    step<<<nc/32, 32, 0, s>>>(regfile, pcfile, gmem, svec);
    core_status_t* svec_h = (core_status_t*)malloc(sizeof(core_status_t) * nc);
    ccE(cudaMemcpyAsync(svec_h, svec, sizeof(core_status_t) * nc, cudaMemcpyDeviceToHost));
    ccE(cudaStreamSynchronize(s));
    for(int i = 0; i < nc; i++) {
        printf("core[%2d]: [%s]\n", i + qnum*nc, CStateToString(svec_h[i].state));
    }
}

int main(int argc, char* argv[]) {
    argh::parser cmdl(argv);
    std::string fpath;
    // np is total # of processes - will be divided over nq Queues
    cmdl("np", 512) >> np;
    cmdl("nq", 4) >> nq;
    cmdl("f", std::string(r0path)) >> fpath;
    uint32_t pc0;
    std::unique_ptr<const Binary> binary = std::unique_ptr<const Binary>{Parser::parse(fpath)};
    pc0 = binary->entrypoint();
    mmc_t mmc;
    // load the binary
    for (const Segment& segment : binary->segments()) {
        uint64_t addr = segment.virtual_address();
        uint64_t vsize = segment.virtual_size();
        uint64_t contentsize = segment.get_content_size();
        printf("[SEG] contentsize: %lu, vsize: %lu starting @: 0x%x\n", contentsize, vsize, (uint32_t)addr);
        if(segment.type() != SEGMENT_TYPES::PT_LOAD) continue;
        if(segment.has(".cuda_constant")){
            cuda_constant_seg = segment.content().data();
            printf("\t[.cuda_constant] \n");
            assert(addr == 0);
            if(vsize < CRO_MAX_SIZE){
                init_cmem((void*)cuda_constant_seg, vsize);
                cmem_off = vsize;
                set_cro(vsize);
            } else {
                cro_spill = vsize - CRO_MAX_SIZE;
                printf("CRO spill: %#lx\n", (vsize - CRO_MAX_SIZE));
                init_cmem((void*)segment.content().data(), CRO_MAX_SIZE);
                cmem_off = CRO_MAX_SIZE;
                set_cro(CRO_MAX_SIZE);
                // currently we don't share the spill region, every core gets its own copy
            }
            continue;
        }
        if(segment.has(".cuda_global")){
            printf("\t[.cuda_global] \n");
            #ifndef ILV
            mmc.push_back(mts_t(addr - cmem_off, vsize, segment.content().data()));
            #else
            ccE(cudaMalloc(gmembase, vsize));
            ccE(cudaMemcpy(gmembase, segment.content().data(), vsize, cudaMemcpyHostToDevice));
            #endif
            continue;
        }    
    }
    printf("%s\n", banner);
    printf("|np: %d|nq: %d| f: %s|\n",np, nq, fpath.c_str());
    cstreams = new cudaStream_t[nq];
    int32_t mpc = CORE_GMEM_SIZE;
    set_cms(&mpc);
    ccE(cudaMallocManaged(&gmem_pool, CORE_GMEM_SIZE * np));
    using clock = std::chrono::system_clock;
    const auto t0 = clock::now();
    for(int i = 0; i < nq; i++){
        ccE(cudaStreamCreate(cstreams+i));
        hths.emplace_back(tX, cstreams[i], np/nq, mpc, pc0, mmc, i);
    }
    for(int i = 0; i < nq; i++){
        hths[i].join();
    }
    const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
    double ke = (double) np/ (double) 1000.0;
    double rate = (ke / (double)dt) * 1000000.0;
    printf("dt: %f Kexecs in %f (ms) : %f (Ke/s)\n", ke ,(double) dt / (double) 1000.0, rate);
}