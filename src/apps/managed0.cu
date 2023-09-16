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
#include <mutex>
#include <stack>
#include <LIEF/ELF.hpp>
#include <LIEF/logging.hpp>
#include <sys/stat.h>
#define CORE_GMEM_SIZE (size_t)1024 * 1024 * 16 * 8
int32_t mpc = CORE_GMEM_SIZE;
uint32_t pc0;

using namespace LIEF::ELF;
const char *r0path = "./ta/bin/hello";
extern void initialize(REG *&regfile, REG *&pcfile, core_status_t *&svec, cudaStream_t *streams, int np, int nq);
extern void set_cms(int32_t *);
extern int32_t calc_mpc(int np, int nq);
extern void set_cro(uint32_t cro = 0);
extern void init_cmem(void *, uint32_t);
std::vector<std::thread> hths;
cudaStream_t *cstreams;
const uint8_t *cuda_constant_seg;

typedef std::tuple<uint32_t, uint32_t, const uint8_t *> mts_t;
typedef std::vector<mts_t> mmc_t;
bool print_final_status = false;

uint32_t cmem_off = 0;
uint32_t cro_spill = 0;
uint8_t *gmembase = NULL;
uint32_t np, nq, nc;
uint8_t *gmem_pool;
std::mutex mtx;

struct core_handle
{
    REG *regfile;
    REG *pcfile;
    uint8_t *gmem;
};

core_handle *chandles;

std::unordered_map<uint32_t, std::string> symmap;
#define DEFAULT_FSDIR "./fs"

using stdout_info_t = std::pair<int, FILE *>;
stdout_info_t *stdout_files;

void setup_stdout(int idx0, int nc)
{
    // For each core set up a file as stdout: p<i>.stdout
    for (int j = 0; j < nc; j++)
    {
        int i = idx0 + j;
        std::string fname = std::string(DEFAULT_FSDIR) + "/p" + std::to_string(i) + ".stdout";
        FILE *fp = fopen(fname.c_str(), "w");
        assert(fp != NULL);
        int fd = fileno(fp);
        stdout_files[i] = stdout_info_t(fd, fp);
    }
}

bool handle_scall(core_status_t &cstate, int coreid, cudaStream_t s)
{
    REG *regfile_h = (REG *)malloc(sizeof(REG) * NUM_OF_REGS);
    ccE(cudaMemcpyAsync(regfile_h, chandles[coreid].regfile, sizeof(REG) * NUM_OF_REGS, cudaMemcpyDeviceToHost, s));
    ccE(cudaStreamSynchronize(s));
    switch (regfile_h[17])
    {
    case SYS_fstat:
    {
        int32_t fd = regfile_h[10];
        if (fd != 1)
        {
            printf("Unhandled fd: %d\n", fd);
            return false;
        }
        struct stat *st = (struct stat *)malloc(sizeof(struct stat));
        int ret = fstat(stdout_files[coreid].first, st);
        ccE(cudaMemcpyAsync(chandles[coreid].regfile + 10, &ret, sizeof(int), cudaMemcpyHostToDevice, s));
        if (ret == 0)
        {
            ccE(cudaMemcpyAsync(chandles[coreid].gmem + regfile_h[11], st, sizeof(struct stat), cudaMemcpyHostToDevice, s));
        }
        break;
    }
    case SYS_brk:
    {
        uint32_t addr = regfile_h[10];
        uint32_t ret = -1;
        if (addr == 0)
        {
            // return the current brk
            ret = mpc;
        }
        ccE(cudaMemcpyAsync(chandles[coreid].regfile + 10, &ret, sizeof(int), cudaMemcpyHostToDevice, s));
        break;
    }
    case SYS_write:
    {
        int fd = regfile_h[10];
        uint32_t addr = regfile_h[11];
        uint32_t len = regfile_h[12];
        uint32_t ret = -1;
        if (fd != 1)
        {
            printf("Unhandled fd: %d\n", fd);
            return false;
        }
        char *buf = (char *)malloc(len + 1);
        ccE(cudaMemcpyAsync(buf, chandles[coreid].gmem + (addr - cmem_off), len, cudaMemcpyDeviceToHost, s));
        uint32_t r = fwrite(buf, 1, len, stdout_files[coreid].second);
        printf("Writing to stdout: addr: %#x, len: %#x [%d] | coreid: %d\n", addr, len, r, coreid);
        ret = r;
        ccE(cudaMemcpyAsync(chandles[coreid].regfile + 10, &ret, sizeof(int), cudaMemcpyHostToDevice, s));
        break;
    }
    default:
    {
        return false;
    }
    }
    return true;
}

void tX(cudaStream_t s, mmc_t mmc, int qnum)
{
    setup_stdout(qnum * nc, nc);
    REG *regfile, *pcfile;
    REG *pc0_h = (REG *)malloc(sizeof(REG));
    std::stack<std::string> backtrace;
    core_status_t *svec;
    uint8_t *gmem = gmem_pool + (CORE_GMEM_SIZE * nc * qnum);
    printf("%x\n", gmem_pool[128]);
    ccE(cudaMallocAsync(&regfile, NUM_OF_REGS * sizeof(REG) * nc, s));
    ccE(cudaMallocAsync(&pcfile, sizeof(REG) * nc, s));
    ccE(cudaMallocAsync(&svec, sizeof(core_status_t) * nc, s));
    printf("mmc.size(): %zu\n", mmc.size());
    assert(mmc.size() <= 1);
    mts_t mts = mmc[0];
    uint32_t addr = std::get<0>(mts);
    uint32_t vsize = std::get<1>(mts);
    const uint8_t *gdata = std::get<2>(mts);
    for (int i = 0; i < nc; i++)
    {
        // allocate the spill
        if (cro_spill)
        {
            ccE(cudaMemcpyAsync(gmem + CORE_GMEM_SIZE * i, cuda_constant_seg + CRO_MAX_SIZE, cro_spill, cudaMemcpyDefault, s));
        }
        // printf("[%d] gmem dst: %p, gdata: %p, addr: %#x, vsize: %#x\n", qnum, gmem + CORE_GMEM_SIZE * i + addr, gdata, addr, vsize);
        ccE(cudaMemcpyAsync(gmem + CORE_GMEM_SIZE * i + addr, (void *)gdata, vsize, cudaMemcpyDefault, s));
        chandles[qnum * nc + i].regfile = regfile + NUM_OF_REGS * i;
        chandles[qnum * nc + i].pcfile = pcfile + i;
        chandles[qnum * nc + i].gmem = gmem + CORE_GMEM_SIZE * i;
    }
    // execute
    initPC<<<nc / 32, 32, 0, s>>>(pcfile, pc0);
    initSP<<<nc / 32, 32, 0, s>>>(regfile, CORE_GMEM_SIZE);
    ccE(cudaMemsetAsync(svec, 0x0, sizeof(core_status_t) * nc, s));
    core_status_t *svec_h = (core_status_t *)malloc(sizeof(core_status_t) * nc);
    // ccE(cudaMemcpyAsync(svec_h, svec, sizeof(core_status_t) * nc, cudaMemcpyDeviceToHost));
    ccE(cudaStreamSynchronize(s));
    if (qnum != 0)
        return;
    bool *pop_bt;
    REG* regfile_h = (REG *)malloc(sizeof(REG) * NUM_OF_REGS);
    ccE(cudaMallocManaged(&pop_bt, sizeof(bool)));
    for (int i = 0; i < 1400; i++)
    {
        step<<<nc / 32, 32, 0, s>>>(regfile, pcfile, gmem, svec, 1);
        ccE(cudaMemcpyAsync(svec_h, svec, sizeof(core_status_t) * nc, cudaMemcpyDeviceToHost, s));
        ccE(cudaDeviceSynchronize());
        mtx.lock();
        printf("t[%2d]: <[%s], %#x>\n", i + qnum*nc, CStateToString(svec_h[0].state), svec_h[0].addr);
        dumpS<<<1, 1, 0, s>>>(regfile, pcfile, gmem, svec, 0, pop_bt);
        mtx.unlock();
        ccE(cudaMemcpyAsync(regfile_h, regfile, sizeof(REG) * NUM_OF_REGS, cudaMemcpyDeviceToHost, s));
        ccE(cudaMemcpyAsync(pc0_h, pcfile, sizeof(REG), cudaMemcpyDeviceToHost, s));
        ccE(cudaDeviceSynchronize());

            if (symmap.find(*pc0_h) != symmap.end())
            {
                printf("pushing: %s: a0=%#x, a1=%#x, a2=%#x, a3=%#x\n", symmap[*pc0_h].c_str(), regfile_h[10], regfile_h[11], regfile_h[12], regfile_h[13]);
                backtrace.push(symmap[*pc0_h]);
            }
    
        for (int i = 0; i < nc; i++)
        {
            if(svec_h[i].state == MAXSTEP) continue;
            if(svec_h[i].state == ECALL) {
                printf("HANDLING SYSCALL\n");
                if(!handle_scall(svec_h[i], qnum*nc + i, s)) {
                    printf("Unhandled syscall!\n");
                    mtx.unlock();
                    return;
                }
                svec_h[i].state = ERTN;
                ccE(cudaMemcpyAsync(svec + i, &svec_h[i], sizeof(core_status_t), cudaMemcpyHostToDevice, s));
            }
        }
    }
}

int main(int argc, char *argv[])
{
    argh::parser cmdl(argv);
    std::string fpath;
    // np is total # of processes - will be divided over nq Queues
    cmdl("np", 64) >> np;
    cmdl("nq", 2) >> nq;
    cmdl("f", std::string(r0path)) >> fpath;
    nc = np / nq;
    assert(np % nq == 0);
    chandles = (core_handle *)malloc(sizeof(core_handle) * np);
    mkdir(DEFAULT_FSDIR, 0777);
    stdout_files = new stdout_info_t[np];
    std::unique_ptr<const Binary> binary = std::unique_ptr<const Binary>{Parser::parse(fpath)};
    pc0 = binary->entrypoint();
    mmc_t mmc;
    // load the binary
    LIEF::ELF::Binary::it_const_symbols symbols = binary->symbols();
    for (const Symbol &symbol : symbols)
    {
        if (symbol.type() == LIEF::ELF::ELF_SYMBOL_TYPES::STT_FUNC)
        {
            printf("symbol: %s, addr: %#x\n", symbol.name().c_str(), symbol.value());
            symmap[symbol.value()] = symbol.name();
        }
    }
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
            cuda_constant_seg = segment.content().data();
            printf("\t[.cuda_constant] \n");
            assert(addr == 0);
            if (vsize < CRO_MAX_SIZE)
            {
                init_cmem((void *)cuda_constant_seg, vsize);
                cmem_off = vsize;
                set_cro(vsize);
            }
            else
            {
                cro_spill = vsize - CRO_MAX_SIZE;
                printf("CRO spill: %#lx\n", (vsize - CRO_MAX_SIZE));
                init_cmem((void *)segment.content().data(), CRO_MAX_SIZE);
                cmem_off = CRO_MAX_SIZE;
                set_cro(CRO_MAX_SIZE);
                // currently we don't share the spill region, every core gets its own copy
            }
            continue;
        }
        if (segment.has(".cuda_global"))
        {
            printf("\t[.cuda_global] \n");
            printf("last addr: %#x\n", addr + vsize);
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
    set_cms(&mpc);
    ccE(cudaMallocManaged(&gmem_pool, CORE_GMEM_SIZE * np));
    using clock = std::chrono::system_clock;
    const auto t0 = clock::now();
    for (int i = 0; i < nq; i++)
    {
        ccE(cudaStreamCreate(cstreams + i));
        hths.emplace_back(tX, cstreams[i], mmc, i);
    }
    for (int i = 0; i < nq; i++)
    {
        hths[i].join();
    }
    const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
    double ke = (double)np / (double)1000.0;
    double rate = (ke / (double)dt) * 1000000.0;
    printf("dt: %f Kexecs in %f (ms) : %f (Ke/s)\n", ke, (double)dt / (double)1000.0, rate);
}