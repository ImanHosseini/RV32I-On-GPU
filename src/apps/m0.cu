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
#include <string>
#include <unordered_set>
#include "fmt/core.h"
#include "fmt/color.h"
// using clock = std::chrono::system_clock;

#include <LIEF/ELF.hpp>
#include <LIEF/logging.hpp>
#include <disman.hpp>

using namespace LIEF::ELF;
const char *r0path = "./ta/bin/m0";

// extern __global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep);
extern void initialize(REG *&regfile, REG *&pcfile, core_status_t *&svec, cudaStream_t *streams, int np, int nq);
extern void set_cms(int32_t *);
extern int32_t calc_mpc(int np, int nq);
extern void set_cro(uint32_t cro = 0);
extern void init_cmem(void *, uint32_t);
std::vector<std::thread> hths;
cudaStream_t *cstreams;

typedef std::tuple<uint32_t, uint32_t, const uint8_t *> mts_t;
typedef std::vector<mts_t> mmc_t;

uint32_t cmem_off = 0;
uint8_t *gmembase = NULL;

enum CMND_T
{
    QUIT,
    STEP,
    CONT,
    RINFO,
    SET_BKPT,
    CLEAR_BKPT,
    SHOW_BKPT,
    SHOW_DISAS
};
std::map<std::string, CMND_T> s2CMND = {
    {"s", STEP},
    {"q", QUIT},
    {"c", CONT},
    {"ri", RINFO},
    {"b", SET_BKPT},
    {"bc", CLEAR_BKPT},
    {"bi", SHOW_BKPT},
    {"ii", SHOW_DISAS}
};

int updatebkpts(std::unordered_set<uint32_t> &bkpts, uint32_t* d_bkpts){
    int n = bkpts.size();
    if(n > 0){
        uint32_t* h_bkpts = new uint32_t[n];
        std::copy(bkpts.begin(), bkpts.end(), h_bkpts);
        ccE(cudaFree(d_bkpts));
        ccE(cudaMalloc(&d_bkpts, sizeof(uint32_t) * n));
        ccE(cudaMemcpy(d_bkpts, h_bkpts, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));
    }else{

    }
    return n;
}

void printREGS(REG *regfile, REG *pcfile)
{
    int regfile_size = NUM_OF_REGS * sizeof(REG);
    int pc_size = sizeof(REG);
    REG *regfile_h = (REG *)malloc(regfile_size);
    REG pc = 0x0;
    ccE(cudaMemcpy(regfile_h, regfile, regfile_size, cudaMemcpyDeviceToHost));
    ccE(cudaMemcpy(&pc, pcfile, pc_size, cudaMemcpyDeviceToHost));
    {
        // cout << fg::green << "PC" << fg::gray << " : " << fg::yellow << "0x" << fg::blue << setw(8) << hex  << pc << fg::reset << endl;
        fmt::print(fg(fmt::color::green), "PC");
        fmt::print(" : ");
        fmt::print(fg(fmt::color::dark_orange), "{:#010X}\n", pc);
        auto printreg = [&](auto i) {
            fmt::print(fg(fmt::color::green), "R{:<2}", i);
            fmt::print(": ");
            fmt::print(fg(fmt::color::blue), "{:#010X}", regfile_h[i]);
        };
        for (int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 8; j++){
                printreg(8*i + j);
                fmt::print("|");
            }
            fmt::print("\n");
        }
    }
}

void tX(cudaStream_t s, uint32_t nc, uint32_t mpc, uint32_t pc0, mmc_t mmc)
{
    REG *regfile, *pcfile;
    core_status_t *svec;
    uint8_t *gmem;
    // int32_t fm = dumpM();
    ccE(cudaMallocAsync(&regfile, NUM_OF_REGS * sizeof(REG) * nc, s));
    ccE(cudaMallocAsync(&pcfile, sizeof(REG) * nc, s));
    // fm = dumpM();
    // printf("DELTA: %d\n", fm - (nc*mpc));
    ccE(cudaMallocAsync(&svec, sizeof(core_status_t) * nc, s));
    // uint32_t fm = dumpM();
    // printf("DELTA: %d\n", free_memory - (np*mpc));
    // MPC * NC ~ TOTAL_VRAM / NQ can potentitally be > 4 GB!!
    ccE(cudaMallocAsync(&gmem, (uint64_t)mpc * nc, s));
    // initPC<<<nc/32, 32, 0, s>>>(pcfile, pc0);
    // initSP<<<nc/32, 32, 0, s>>>(regfile, mpc + cmem_off);
    // prepare the memory
    for (mts_t mts : mmc)
    {
        for (int i = 0; i < nc; i++)
        {
            ccE(cudaMemcpyAsync(gmem + i * mpc + std::get<0>(mts), std::get<2>(mts), std::get<1>(mts), cudaMemcpyHostToDevice, s));
        }
    }
    // set up the cores
    initPC<<<nc / 32, 32, 0, s>>>(pcfile, pc0);
    initSP<<<nc / 32, 32, 0, s>>>(regfile, mpc + cmem_off);
    ccE(cudaMemsetAsync(svec, 0x0, sizeof(core_status_t) * nc, s));
    int n_bkpts = 0;
    std::unordered_set<uint32_t> bkpts;
    uint32_t* d_bkpts = 0;
    while (true)
    {
        // std::cout << rang::style::bold << rang::fg::red << "GRDBG>" << rang::style::reset;
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "GRDBG>");
        std::string cmnd;
        getline(std::cin, cmnd);
        std::vector<std::string> toks = split(std::string(cmnd));
        int tlen = toks.size();
        if (s2CMND.find(toks[0]) == s2CMND.end())
        {
            continue;
        }
        // printf("TLEN: %d\n", tlen);
        // for(auto tok : toks){
        //     printf("TOK: %s\n", tok.c_str());
        // }
        switch (s2CMND[toks[0]])
        {
        case STEP:
        {
            if (tlen == 1)
            {
                step<<<nc / 32, 32, 0, s>>>(regfile, pcfile, gmem, svec, 1, d_bkpts, n_bkpts);
            }
            else
            {
                int steps = 1;
                try
                {
                    steps = stoi(toks[1]);
                }
                catch (...)
                {
                }
                step<<<nc / 32, 32, 0, s>>>(regfile, pcfile, gmem, svec, steps, d_bkpts, n_bkpts);
            }
            break;
        }
        case QUIT:
        {
            exit(0);
            break;
        }
        case CONT:
        {
            step<<<nc / 32, 32, 0, s>>>(regfile, pcfile, gmem, svec, 0, d_bkpts, n_bkpts);
            break;
        }
        case RINFO:{
            printREGS(regfile, pcfile);
            break;
        }
        case SET_BKPT:{
            if(tlen == 1){
                fmt::print("No breakpoint specified\n");
            }
            else{
                try{
                    uint32_t bkpt = stoul(toks[1], nullptr, 16);
                    bkpts.insert(bkpt);
                    n_bkpts = updatebkpts(bkpts, d_bkpts);
                }
                catch(...){
                    fmt::print("Invalid breakpoint specified\n");
                }
            }
            break;
        }
        case CLEAR_BKPT:{
            if(tlen == 1){
                bkpts.clear();
                n_bkpts = updatebkpts(bkpts, d_bkpts);
            }
            else{
                try{
                    uint32_t bkpt = stoul(toks[1], nullptr, 16);
                    if(bkpts.find(bkpt) != bkpts.end()){
                        bkpts.erase(bkpt);
                        n_bkpts = updatebkpts(bkpts, d_bkpts);
                    } else {
                        fmt::print("Breakpoint {} not found\n", fmt::format("{:#x}", bkpt));
                    }
                }
                catch(...){
                    fmt::print("Invalid breakpoint specified\n");
                }
            }
            break;
        }
        case SHOW_BKPT:{
            if(n_bkpts == 0){
                fmt::print("No breakpoints set\n");
            } else {
                for(auto bkpt : bkpts){
                    fmt::print("bp: {}\n", fmt::format("{:#x}", bkpt));
                }
            }
        }
        default:
        {
            break;
        }
        }
        ccE(cudaStreamSynchronize(s));
    }
}

int main(int argc, char *argv[])
{
    argh::parser cmdl(argv);
    int np;
    int nq = 1;
    std::string fpath;
    // np is total # of processes - will be divided over nq Queues
    cmdl("np", 32) >> np;
    cmdl("f", std::string(r0path)) >> fpath;

    uint32_t pc0;
    std::unique_ptr<const Binary> binary = std::unique_ptr<const Binary>{Parser::parse(fpath)};
    pc0 = binary->entrypoint();
    mmc_t mmc;
    // Load up the binary
    for (const Segment &segment : binary->segments())
    {
        uint64_t addr = segment.virtual_address();
        uint64_t vsize = segment.virtual_size();
        uint64_t contentsize = segment.get_content_size();
        printf("[SEG] contentsize: %lu, vsize: %lu starting @: 0x%x\n", contentsize, vsize, (uint32_t)addr);
        if (segment.type() != SEGMENT_TYPES::PT_LOAD)
            continue;
        if (segment.has(".cuda_constant"))
        {
            printf("\t[.cuda_constant] \n");
            assert(addr == 0);
            assert(vsize < CRO_MAX_SIZE);
            init_cmem((void *)segment.content().data(), vsize);
            cmem_off = vsize;
            set_cro(vsize);
            continue;
        }
        if (segment.has(".cuda_global"))
        {
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
    printf("|np: %d|nq: %d| f: %s|\n", np, nq, fpath.c_str());
    cstreams = new cudaStream_t[nq];
    int32_t mpc = calc_mpc(np, nq);
    set_cms(&mpc);
    for (int i = 0; i < nq; i++)
    {
        ccE(cudaStreamCreate(cstreams + i));
        hths.emplace_back(tX, cstreams[i], np / nq, mpc, pc0, mmc);
    }
    for (int i = 0; i < nq; i++)
    {
        hths[i].join();
    }
}