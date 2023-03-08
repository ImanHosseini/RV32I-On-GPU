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
#include <deque>
#include <mutex>
#include <uiutils.h>
#include <fmt/core.h>
#include <condition_variable>
#include <exception>

// using clock = std::chrono::system_clock;
std::condition_variable cv;
std::mutex wmut;
std::vector<int> tstep;
REG sel_regs[32] = {0};
int sel_id = 0;
int sel_qid = 0;
int sel_cid = 0;

struct Qdata{
    core_status_t* svec_h = NULL;
    REG* pcfile_h = NULL;
    int halted = 0;
};
std::vector<Qdata> qDs;

#include <LIEF/ELF.hpp>
#include <LIEF/logging.hpp>

using namespace LIEF::ELF;
const char *r0path = "./ta/bpff/bpff";

extern void initialize(REG *&regfile, REG *&pcfile, core_status_t *&svec, cudaStream_t *streams, int np, int nq);
extern void set_cms(int32_t *);
extern int32_t calc_mpc(int np, int nq);
extern void set_cro(uint32_t cro = 0);
extern void init_cmem(void *, uint32_t);
std::vector<std::thread> hths;
cudaStream_t *cstreams;

typedef std::tuple<uint32_t, uint32_t, const uint8_t *> mts_t;
typedef std::vector<mts_t> mmc_t;
bool print_final_status = false;

uint32_t cmem_off = 0;
char cmnd[200];
int np, nq;


void handle_cmnd(){
    std::vector<std::string> tokens = split(std::string(cmnd));
    if(tokens[0] == "s"){
        int steps = 1;
        if(tokens.size() > 1){
            try{
                steps = std::stoi(tokens[1]);
            }catch (const std::exception& e){
                // error msg?
            }
        }
        {
            std::lock_guard lk(wmut);
            for(int i = 0; i < nq; i++){
                tstep[i] = steps;
            }
        }
        cv.notify_all();
    }
}

static int qb_lvl = 0;
static int qb_selq = 0;
static int qb_selb = 0;

static bool qb(int i){
    return ImGui::QBtn(fmt::format("Q{}\n<<<{}x{}>>>\nHalts:{}/{}\n", i, (np/nq)/32, 32, qDs[i].halted, np/nq).c_str(), ImVec2(12,6), ImVec4(ImColor(0, 200, 0)));
}

static bool qb_b(int i){
    return ImGui::QBtn(fmt::format("B{}\nC[{}-{}]", i, qb_selq*(np/nq) + i*32, qb_selq*(np/nq) + (i+1)*32).c_str(), ImVec2(12,6), ImVec4(ImColor(0, 200, 0)));
}

static bool qb_w(int i){
    return ImGui::QBtn(fmt::format("th{}:C{}\nPC:0x{:08x}", i, qb_selq*(np/nq)+qb_selb*32+i,qDs[qb_selq].pcfile_h[qb_selb*32+i]).c_str(), ImVec2(14,4), ImVec4(ImColor(0, 200, 0)));
}

static void reg(int i, bool sl = true){
    // tY(fmt::format("X{:02}:0x{:08x}",i,sel_regs[i]).c_str());
    if(sl) tR("|");
    tY(fmt::format("x{:02}",i).c_str());
    ImGui::SameLine();
    tW(":"); ImGui::SameLine();
    tG(fmt::format("0x{:08x}", sel_regs[i]).c_str(), true);
    tR("|", sl);
}

// ImGui::GetWindowDrawList()->AddRect(wPos, { wPos.x + wSize.x, wPos.y + wSize.y }, ImColor(1.f, 1.f, 1.f, 1.f), 20.f);

static void render_regs(){
    
    // ImGui::PushStyleColor(ImGuiCol_ChildBg, ImU32(ImColor(30,30,120)));
    static ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
    // ImGui::SameLine();
    // ImGui::BeginChild("REGS", ImVec2(17*2-1,17), true, flags);
    tR("REGISTERS", false);
    for(int i = 0; i < 32; i++){
        reg(i, i % 2 == 0);
    }
    // ImGui::EndChild();
    // ImGui::PopStyleColor();
}

static void show()
{
    static bool use_work_area = false;
    static ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDecoration;

    // We demonstrate using the full viewport area or the work area (without menu-bars, task-bars etc.)
    // Based on your use case you may want one of the other.
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
    ImVec2 vs = use_work_area ? viewport->WorkSize : viewport->Size;
    // vs.x -= 5.0;
    ImGui::SetNextWindowSize(vs);
    // ImGui::SetNextWindowSize(ImVec2(50.0,50.0));
    if (ImGui::Begin("FS window", NULL, flags))
    {
        ImGui::TextColored(ImVec4(ImColor(255, 0, 0)), "[grdbg v0.1]");
        tG(std::to_string(np).c_str()); tW(" cores: "); tG(std::to_string(nq).c_str()); tW(" Queues of "); tG(fmt::format("{} = <<<{},32>>>",(np/nq),(np/nq)/32).c_str(), false);
        if(viewport->Size.x < 80.0f || viewport->Size.y < 24.0f){
            ImGui::Text("Screen too smol!");
            ImGui::End();
            return;
        }
        static const char* tabNames[] = {"Fleet", "Command"};
        static const int numTabs = 2;
        static int selectedTab = 0;
        ImGui::TabLabels(numTabs,tabNames,selectedTab);
        ImGui::Separator();
        switch(selectedTab){
            case 0:{
                if(ImGui::Button("back")){
                    if(qb_lvl > 0){
                        qb_lvl--;
                    }
                }
                switch(qb_lvl){
                    case 0:{
                        for(int i = 0; i < nq; i++){
                        if(qb(i)){
                            qb_lvl++;
                            qb_selq = i;    
                        };
                        ImGui::SameLine();
                        }
                        break;
                    }
                    case 1:{
                        for(int i = 0; i < (np/nq)/32; i++){
                        if(qb_b(i)){
                            qb_lvl++;
                            qb_selq = i;    
                        };
                        if(i % 8 != 7) ImGui::SameLine();
                        }
                        break;
                    }
                    case 2:{
                        for(int i = 0; i < 32; i++){
                        if(qb_w(i)){  
                        };
                        if(i % 4 != 3) ImGui::SameLine();
                        }
                        break;
                    }
                    default: break;
                }
                break;
            }
            case 1:{
                tY("core #"); tR(std::to_string(sel_id).c_str()); tY(" | "); tR(std::to_string(sel_cid).c_str()); tY(" @ Q"); tR(std::to_string(sel_qid).c_str());
                tY(" | "); tB(fmt::format("PC: 0x{:08x}", qDs[sel_qid].pcfile_h[sel_cid]).c_str(), false);
                ImGui::Text(">");
                ImGui::SameLine();
                ImGui::PushItemWidth(-1);
                int f = ImGuiInputTextFlags_EnterReturnsTrue;
                if(ImGui::InputText(" ", cmnd, 200, f)){
                    handle_cmnd();
                    memset(cmnd, 0, 200);
                    ImGui::SetKeyboardFocusHere(-1);
                }
                ImGui::PopItemWidth();
                render_regs();
                break;
            }
            default: break;
        }
    }
    ImGui::End();
}

void uithread(){
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto screen = ImTui_ImplNcurses_Init(true);
    ImTui_ImplText_Init();
    int nframes = 0;
    float fval = 1.23f;
    ImGuiStyle &style = ImGui::GetStyle();
    style.ButtonTextAlign = ImVec2(0.0f, 0.0f);
    while (true)
    {
        ImGui::NewFrame();
        ImTui_ImplNcurses_NewFrame();
        ImTui_ImplText_NewFrame();
        show();
        ImGui::Render();
        ImTui_ImplText_RenderDrawData(ImGui::GetDrawData(), screen);
        ImTui_ImplNcurses_DrawScreen();
    }
    ImTui_ImplText_Shutdown();
    ImTui_ImplNcurses_Shutdown();
}

void tX(cudaStream_t s, uint32_t nc, uint32_t mpc, uint32_t pc0, mmc_t mmc, const int thidx)
{
    REG *regfile, *pcfile;
    core_status_t *svec;
    uint8_t *gmem;
    // int32_t fm = dumpM();
    ccE(cudaMallocAsync(&regfile, NUM_OF_REGS * sizeof(REG) * nc, s));
    ccE(cudaMallocAsync(&pcfile, sizeof(REG) * nc, s));
    REG* _pcfile_h = (REG*)malloc(sizeof(REG)*nc);
    std::fill_n (_pcfile_h, nc, pc0);
    qDs[thidx].pcfile_h = _pcfile_h;
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
    // this is the "packet" -> fill with random? fill with random fast using gpu random? - no.
    mts_t mts = mmc[0];
    uint32_t addr = std::get<0>(mts);
    uint32_t vsize = nc * std::get<1>(mts);
    uint8_t *pkts = new uint8_t[vsize];
    using random_bytes_engine =
        std::independent_bits_engine<std::default_random_engine, CHAR_BIT,
                                     unsigned char>;
    random_bytes_engine rbe(0x37);
    std::generate(pkts, pkts + vsize, std::ref(rbe));
    ccE(cudaMemcpyAsync(gmem + addr, pkts, vsize, cudaMemcpyHostToDevice, s));
    initPC<<<nc / 32, 32, 0, s>>>(pcfile, pc0);
    initSP<<<nc / 32, 32, 0, s>>>(regfile, mpc + cmem_off);
    ccE(cudaMemsetAsync(svec, 0x0, sizeof(core_status_t) * nc, s));
    
    int iter = 0;
    while(true){
        std::unique_lock<std::mutex> lk(wmut);
        cv.wait(lk, [thidx] {return tstep[thidx] > 0;});
        iter += tstep[thidx];
        tstep[thidx] = 0;
        lk.unlock();
        step<<<nc / 32, 32, 0, s>>>(regfile, pcfile, gmem, svec, iter);
        ccE(cudaMemcpy(_pcfile_h, pcfile, sizeof(REG) * nc, cudaMemcpyDeviceToHost));
        if(sel_qid == thidx){
            ccE(cudaMemcpy(sel_regs, regfile + sel_cid * sizeof(REG) * 32, sizeof(REG) * 32, cudaMemcpyDeviceToHost));
        }
    }
    
    // print statuses
    // if (print_final_status)
    // {
    //     core_status_t *svec_h = (core_status_t *)malloc(sizeof(core_status_t) * nc);
    //     ccE(cudaMemcpy(svec_h, svec, sizeof(core_status_t) * nc, cudaMemcpyDeviceToHost));
    //     for (int i = 0; i < nc; i++)
    //     {
    //         printf("[%d]: [%d]\n", i, svec_h[i].state);
    //     }
    // }
    ccE(cudaStreamSynchronize(s));
    delete[] pkts;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl(argv);

    std::string fpath;
    // np is total # of processes - will be divided over nq Queues
    cmdl("np", 128) >> np;
    cmdl("nq", 2) >> nq;
    for(int i = 0; i < nq; i++){
        tstep.push_back(0);
        qDs.push_back(Qdata());
    }
    cmdl("f", std::string(r0path)) >> fpath;
    if (cmdl["fst"])
    {
        print_final_status = true;
    }
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
            mmc.push_back(mts_t(addr - cmem_off, vsize, segment.content().data()));
            continue;
        }
    }
    printf("%s\n", banner);
    // printf("|np: %d|nq: %d| f: %s|\n", np, nq, fpath.c_str());
    cstreams = new cudaStream_t[nq];
    int32_t mpc = calc_mpc(np, nq);
    set_cms(&mpc);
    for (int i = 0; i < nq; i++)
    {
        ccE(cudaStreamCreate(cstreams + i));
        hths.emplace_back(tX, cstreams[i], np / nq, mpc, pc0, mmc, i);
    }

    std::thread uit(uithread);
    for (int i = 0; i < nq; i++)
    {
        hths[i].join();
    }
    uit.join();
}