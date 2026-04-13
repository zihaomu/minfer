//
// Per-layer benchmark profiler for minfer Net.
// Created by Antigravity on 2026/4/3.
//

#ifndef MINFER_BENCHMARK_PROFILER_H
#define MINFER_BENCHMARK_PROFILER_H

#include "minfer/layer.h"
#include "minfer/context.h"

#include <chrono>
#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace minfer
{

struct LayerSubProfile
{
    std::string stageName;

    double prefill_total_us = 0.0;
    int    prefill_calls    = 0;

    double decode_total_us = 0.0;
    int    decode_calls    = 0;

    double forward_total_us = 0.0;
    int    forward_calls    = 0;
};

struct LayerProfile
{
    std::string layerName;
    LayerType   layerType = LayerType::UnSupported;
    int         layerId   = -1;

    // Prefill phase
    double prefill_total_us = 0.0;
    int    prefill_calls    = 0;

    // Decode phase
    double decode_total_us = 0.0;
    int    decode_calls    = 0;

    // Plain forward (no InferenceContext)
    double forward_total_us = 0.0;
    int    forward_calls    = 0;

    std::vector<LayerSubProfile> subProfiles;
};

class BenchmarkProfiler
{
public:
    void resize(int num_layers)
    {
        profiles_.resize(num_layers);
    }

    void setLayerInfo(int layerId, const std::string& name, LayerType type)
    {
        if (layerId < 0 || layerId >= (int)profiles_.size())
            profiles_.resize(layerId + 1);

        profiles_[layerId].layerId   = layerId;
        profiles_[layerId].layerName = name;
        profiles_[layerId].layerType = type;
    }

    void record(int layerId, InferPhase phase, double elapsed_us)
    {
        auto& p = profiles_[layerId];
        if (phase == InferPhase::Prefill)
        {
            p.prefill_total_us += elapsed_us;
            p.prefill_calls++;
        }
        else
        {
            p.decode_total_us += elapsed_us;
            p.decode_calls++;
        }
    }

    void recordForward(int layerId, double elapsed_us)
    {
        auto& p = profiles_[layerId];
        p.forward_total_us += elapsed_us;
        p.forward_calls++;
    }

    void recordSubEvent(int layerId, InferPhase phase, const std::string& stage_name, double elapsed_us)
    {
        if (layerId < 0 || layerId >= static_cast<int>(profiles_.size()))
        {
            profiles_.resize(layerId + 1);
        }

        auto& p = profiles_[layerId];
        auto it = std::find_if(p.subProfiles.begin(),
                               p.subProfiles.end(),
                               [&](const LayerSubProfile& sp) { return sp.stageName == stage_name; });
        if (it == p.subProfiles.end())
        {
            p.subProfiles.push_back(LayerSubProfile());
            it = p.subProfiles.end() - 1;
            it->stageName = stage_name;
        }

        if (phase == InferPhase::Prefill)
        {
            it->prefill_total_us += elapsed_us;
            it->prefill_calls++;
        }
        else
        {
            it->decode_total_us += elapsed_us;
            it->decode_calls++;
        }
    }

    void recordForwardSubEvent(int layerId, const std::string& stage_name, double elapsed_us)
    {
        if (layerId < 0 || layerId >= static_cast<int>(profiles_.size()))
        {
            profiles_.resize(layerId + 1);
        }

        auto& p = profiles_[layerId];
        auto it = std::find_if(p.subProfiles.begin(),
                               p.subProfiles.end(),
                               [&](const LayerSubProfile& sp) { return sp.stageName == stage_name; });
        if (it == p.subProfiles.end())
        {
            p.subProfiles.push_back(LayerSubProfile());
            it = p.subProfiles.end() - 1;
            it->stageName = stage_name;
        }

        it->forward_total_us += elapsed_us;
        it->forward_calls++;
    }

    void reset()
    {
        for (auto& p : profiles_)
        {
            p.prefill_total_us = 0.0;
            p.prefill_calls    = 0;
            p.decode_total_us  = 0.0;
            p.decode_calls     = 0;
            p.forward_total_us = 0.0;
            p.forward_calls    = 0;
            for (auto& sp : p.subProfiles)
            {
                sp.prefill_total_us = 0.0;
                sp.prefill_calls    = 0;
                sp.decode_total_us  = 0.0;
                sp.decode_calls     = 0;
                sp.forward_total_us = 0.0;
                sp.forward_calls    = 0;
            }
        }
    }

    const std::vector<LayerProfile>& profiles() const { return profiles_; }

    std::string reportString() const
    {
        std::ostringstream os;

        bool has_prefill = false, has_decode = false, has_forward = false;
        for (const auto& p : profiles_)
        {
            if (p.prefill_calls > 0) has_prefill = true;
            if (p.decode_calls  > 0) has_decode  = true;
            if (p.forward_calls > 0) has_forward = true;
        }

        os << "\n";
        os << "╔══════════════════════════════════════════════════════════════════════════╗\n";
        os << "║                     Per-Layer Benchmark Report                          ║\n";
        os << "╚══════════════════════════════════════════════════════════════════════════╝\n";

        if (has_prefill) appendPhase(os, "Prefill", PhaseKind::Prefill);
        if (has_decode)  appendPhase(os, "Decode",  PhaseKind::Decode);
        if (has_forward) appendPhase(os, "Forward", PhaseKind::Forward);

        if (!has_prefill && !has_decode && !has_forward)
        {
            os << "  (no profiling data collected)\n\n";
        }

        return os.str();
    }

    void printReport() const
    {
        const std::string report = reportString();
        std::fwrite(report.data(), 1, report.size(), stdout);
    }

private:
    enum class PhaseKind { Prefill, Decode, Forward };

    static double phaseTotalUs(const LayerProfile& p, PhaseKind kind)
    {
        return (kind == PhaseKind::Prefill) ? p.prefill_total_us :
               (kind == PhaseKind::Decode)  ? p.decode_total_us  :
                                              p.forward_total_us;
    }

    static int phaseCalls(const LayerProfile& p, PhaseKind kind)
    {
        return (kind == PhaseKind::Prefill) ? p.prefill_calls :
               (kind == PhaseKind::Decode)  ? p.decode_calls  :
                                              p.forward_calls;
    }

    static double phaseTotalUs(const LayerSubProfile& p, PhaseKind kind)
    {
        return (kind == PhaseKind::Prefill) ? p.prefill_total_us :
               (kind == PhaseKind::Decode)  ? p.decode_total_us  :
                                              p.forward_total_us;
    }

    static int phaseCalls(const LayerSubProfile& p, PhaseKind kind)
    {
        return (kind == PhaseKind::Prefill) ? p.prefill_calls :
               (kind == PhaseKind::Decode)  ? p.decode_calls  :
                                              p.forward_calls;
    }

    static void appendRule(std::ostringstream& os, int count, const char* token)
    {
        for (int i = 0; i < count; ++i)
        {
            os << token;
        }
    }

    void appendPhase(std::ostringstream& os, const char* title, PhaseKind kind) const
    {
        int max_name_len = 5; // "Layer"
        for (const auto& p : profiles_)
        {
            int len = (int)p.layerName.size();
            if (len > max_name_len) max_name_len = len;
        }
        max_name_len = std::min(max_name_len, 30);

        double total_us = 0.0;
        for (const auto& p : profiles_)
        {
            total_us += phaseTotalUs(p, kind);
        }

        os << "\n  ── " << title << " ──\n";
        os << "  " << std::left << std::setw(max_name_len) << "Layer"
           << " │ " << std::setw(6) << "Calls"
           << " │ " << std::setw(11) << "Total(ms)"
           << " │ " << std::setw(10) << "Avg(ms)"
           << " │ " << std::setw(6) << "%" << "\n";
        os << "  ";
        appendRule(os, max_name_len, "─");
        os << "─┼────────┼─────────────┼────────────┼────────\n";

        for (const auto& p : profiles_)
        {
            const double t_us = phaseTotalUs(p, kind);
            const int calls = phaseCalls(p, kind);

            if (calls == 0) continue;

            double t_ms  = t_us / 1000.0;
            double avg   = t_ms / calls;
            double pct   = (total_us > 0.0) ? (t_us / total_us * 100.0) : 0.0;

            os << "  " << std::left << std::setw(max_name_len) << p.layerName
               << " │ " << std::right << std::setw(6) << calls
               << " │ " << std::setw(9) << std::fixed << std::setprecision(3) << t_ms << "   "
               << " │ " << std::setw(8) << std::fixed << std::setprecision(3) << avg << "   "
               << " │ " << std::setw(5) << std::fixed << std::setprecision(1) << pct << "%\n";
        }

        double total_ms = total_us / 1000.0;
        os << "  ";
        appendRule(os, max_name_len, "─");
        os << "─┼────────┼─────────────┼────────────┼────────\n";
        os << "  " << std::left << std::setw(max_name_len) << "TOTAL"
           << " │        │ " << std::right << std::setw(9) << std::fixed << std::setprecision(3) << total_ms
           << "   │            │ 100.0%\n\n";

        appendSubPhase(os, title, kind);
    }

    void appendSubPhase(std::ostringstream& os, const char* title, PhaseKind kind) const
    {
        bool has_sub_profiles = false;
        int max_name_len = 9; // "Substage"
        for (const auto& p : profiles_)
        {
            for (const auto& sp : p.subProfiles)
            {
                if (phaseCalls(sp, kind) == 0)
                {
                    continue;
                }
                has_sub_profiles = true;
                const int len = static_cast<int>(p.layerName.size() + 1 + sp.stageName.size());
                if (len > max_name_len)
                {
                    max_name_len = len;
                }
            }
        }

        if (!has_sub_profiles)
        {
            return;
        }

        max_name_len = std::min(max_name_len, 48);
        os << "  ── " << title << " Sub-Stages ──\n";
        os << "  " << std::left << std::setw(max_name_len) << "Substage"
           << " │ " << std::setw(6) << "Calls"
           << " │ " << std::setw(11) << "Total(ms)"
           << " │ " << std::setw(10) << "Avg(ms)"
           << " │ " << std::setw(7) << "%Layer" << "\n";
        os << "  ";
        appendRule(os, max_name_len, "─");
        os << "─┼────────┼─────────────┼────────────┼─────────\n";

        for (const auto& p : profiles_)
        {
            const double layer_us = phaseTotalUs(p, kind);
            if (layer_us <= 0.0)
            {
                continue;
            }

            for (const auto& sp : p.subProfiles)
            {
                const int calls = phaseCalls(sp, kind);
                if (calls == 0)
                {
                    continue;
                }

                const double t_us = phaseTotalUs(sp, kind);
                const double t_ms = t_us / 1000.0;
                const double avg = t_ms / calls;
                const double pct = t_us / layer_us * 100.0;
                std::string name = p.layerName + "/" + sp.stageName;
                if (static_cast<int>(name.size()) > max_name_len)
                {
                    name.resize(static_cast<size_t>(max_name_len));
                }

                os << "  " << std::left << std::setw(max_name_len) << name
                   << " │ " << std::right << std::setw(6) << calls
                   << " │ " << std::setw(9) << std::fixed << std::setprecision(3) << t_ms << "   "
                   << " │ " << std::setw(8) << std::fixed << std::setprecision(3) << avg << "   "
                   << " │ " << std::setw(6) << std::fixed << std::setprecision(1) << pct << "%\n";
            }
        }

        os << "\n";
    }

    std::vector<LayerProfile> profiles_;
};

} // namespace minfer

#endif // MINFER_BENCHMARK_PROFILER_H
