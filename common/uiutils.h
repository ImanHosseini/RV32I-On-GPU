#include "imtui/imtui.h"
#include "imtui/imtui-impl-ncurses.h"
#include <string>
#include <vector>
#include <algorithm>

void tW(const char* s, bool sl = true){
    ImGui::TextColored(ImVec4(ImColor(255, 255, 255)), s);
    if(sl){
        ImGui::SameLine();
    }
}

void tY(const char* s, bool sl = true){
    ImGui::TextColored(ImVec4(ImColor(255, 255, 0)), s);
    if(sl){
        ImGui::SameLine();
    }
}

void tR(const char* s, bool sl = true){
    ImGui::TextColored(ImVec4(ImColor(255, 0, 0)), s);
    if(sl){
        ImGui::SameLine();
    }
}

void tG(const char* s, bool sl = true){
    ImGui::TextColored(ImVec4(ImColor(0, 255, 0)), s);
    if(sl){
        ImGui::SameLine();
    }
}

void tB(const char* s, bool sl = true){
    ImGui::TextColored(ImVec4(ImColor(0, 0, 255)), s);
    if(sl){
        ImGui::SameLine();
    }
}

namespace ImGui
{

    IMGUI_API bool TabLabels(int numTabs, const char **tabLabels, int &selectedIndex)
    {
        ImGuiStyle &style = ImGui::GetStyle();

        const ImVec2 itemSpacing = style.ItemSpacing;
        const ImVec4 color = style.Colors[ImGuiCol_Button];
        const ImVec4 colorActive = ImVec4(ImColor(0, 150, 0));
        const ImVec4 colorHover = ImVec4(ImColor(0, 60, 0));
        style.ItemSpacing.x = 0;
        style.ItemSpacing.y = 0;

        if (numTabs > 0 && (selectedIndex < 0 || selectedIndex >= numTabs))
            selectedIndex = 0;

        // Parameters to adjust to make autolayout work as expected:----------
        // The correct values are probably the ones in the comments, but I took some margin so that they work well
        // with a (medium size) vertical scrollbar too [Ok I should detect its presence and use the appropriate values...].
        const float btnOffset = 2.f * style.FramePadding.x;         // [2.f*style.FramePadding.x] It should be: ImGui::Button(text).size.x = ImGui::CalcTextSize(text).x + btnOffset;
        const float sameLineOffset = 2.f * style.ItemSpacing.x;     // [style.ItemSpacing.x]      It should be: sameLineOffset = ImGui::SameLine().size.x;
        const float uniqueLineOffset = 2.f * style.WindowPadding.x; // [style.WindowPadding.x]    Width to be sutracted by windowWidth to make it work.
        //--------------------------------------------------------------------

        float windowWidth = 0.f, sumX = 0.f;

        bool selection_changed = false;
        for (int i = 0; i < numTabs; i++)
        {
            // push the style
            if (i == selectedIndex)
            {
                style.Colors[ImGuiCol_Button] = colorActive;
                style.Colors[ImGuiCol_ButtonActive] = colorActive;
                style.Colors[ImGuiCol_ButtonHovered] = colorActive;
            }
            else
            {
                style.Colors[ImGuiCol_Button] = color;
                style.Colors[ImGuiCol_ButtonActive] = colorActive;
                style.Colors[ImGuiCol_ButtonHovered] = colorHover;
            }

            ImGui::PushID(i); // otherwise two tabs with the same name would clash.
            if (i != 0)
            {
                ImGui::SameLine();
                ImGui::Text("\\");
                ImGui::SameLine();
            }

            // Draw the button
            if (ImGui::Button(tabLabels[i]))
            {
                selection_changed = (selectedIndex != i);
                selectedIndex = i;
            }
            ImGui::PopID();
        }

        // Restore the style
        style.Colors[ImGuiCol_Button] = color;
        style.Colors[ImGuiCol_ButtonActive] = colorActive;
        style.Colors[ImGuiCol_ButtonHovered] = colorHover;
        style.ItemSpacing = itemSpacing;

        return selection_changed;
    }

    IMGUI_API bool QBtn(const char *label, const ImVec2 &size, const ImVec4 color_)
    {
        ImGuiStyle &style = ImGui::GetStyle();
        const ImVec4 _color = style.Colors[ImGuiCol_Button];
        const ImVec2 itemSpacing = style.ItemSpacing;
        const ImVec4 color = color_;
        // const ImVec4 colorActive =  ImVec4(ImColor(0, 150, 0));
        // const ImVec4 colorHover =   ImVec4(ImColor(0, 60, 0));
        style.ItemSpacing.x = 0;
        style.ItemSpacing.y = 0;
        {
            style.Colors[ImGuiCol_Button] = color;
            // style.Colors[ImGuiCol_ButtonActive] =   colorActive;
            // style.Colors[ImGuiCol_ButtonHovered] =  colorHover;
        }
        // ImGui::PushID(i);   // otherwise two tabs with the same name would clash.
        bool ret = ImGui::Button(label, size);
        // ImGui::PopID();

        // // Restore the style
        style.Colors[ImGuiCol_Button] = _color;
        // style.Colors[ImGuiCol_ButtonActive] =   colorActive;
        // style.Colors[ImGuiCol_ButtonHovered] =  colorHover;
        // style.ItemSpacing =                     itemSpacing;
        return ret;
    }
} // namespace ImGui