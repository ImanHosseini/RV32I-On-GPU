#include <iostream>
#include <memory>
#include <stdio.h>
#include <util.cuh>
#include <argh.h>
#include <thread>
#include <vector>
#include <tuple>
#include <random>

#include <LIEF/ELF.hpp>
#include <LIEF/logging.hpp>

using namespace LIEF::ELF;
const char* r0path = "./ta/bpff/bpff";

// extern __global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep);
extern void initialize(REG* &regfile, REG* &pcfile, core_status_t* &svec, cudaStream_t* streams, int np, int nq);
extern void set_cms(int32_t*);
extern int32_t calc_mpc(int np, int nq);
extern void set_cro(uint32_t cro = 0);
extern void init_cmem(void*, uint32_t);
std::vector<std::thread> hths;
cudaStream_t* cstreams;

typedef std::tuple<uint32_t, uint32_t, const uint8_t*> mts_t; 
typedef std::vector<mts_t> mmc_t;
bool print_final_status = false;

uint32_t cmem_off = 0;
uint8_t* gmembase = NULL;

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
    initSP<<<nc/32, 32, 0, s>>>(regfile, mpc + cmem_off);
    // prepare the memory
    // this is the "packet" -> fill with random? fill with random fast using gpu random? - no. 
    mts_t mts = mmc[0];
    uint32_t addr = std::get<0>(mts);
    uint32_t vsize = nc * std::get<1>(mts);
    uint8_t* pkts = new uint8_t[vsize];
    using random_bytes_engine =
    std::independent_bits_engine<std::default_random_engine, CHAR_BIT,
                                 unsigned char>;
    random_bytes_engine rbe;
    std::generate(pkts, pkts+vsize, std::ref(rbe));
    ccE(cudaMemcpyAsync(gmem + addr, pkts, vsize, cudaMemcpyHostToDevice, s));
    delete[] pkts;
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
    // np is total # of processes - will be divided over nq Queues
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
        uint64_t addr = segment.virtual_address();
        uint64_t vsize = segment.virtual_size();
        uint64_t contentsize = segment.get_content_size();
        printf("[SEG] contentsize: %lu, vsize: %lu starting @: 0x%x\n", contentsize, vsize, (uint32_t)addr);
        if(segment.type() != SEGMENT_TYPES::PT_LOAD) continue;
        if(segment.has(".cuda_constant")){
            printf("\t[.cuda_constant] \n");
            assert(addr == 0);
            assert(vsize < CRO_MAX_SIZE);
            init_cmem((void*)segment.content().data(), vsize);
            cmem_off = vsize;
            set_cro(vsize);
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