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

namespace minfer
{

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
        }
    }

    const std::vector<LayerProfile>& profiles() const { return profiles_; }

    void printReport() const
    {
        // Determine which phases have data
        bool has_prefill = false, has_decode = false, has_forward = false;
        for (const auto& p : profiles_)
        {
            if (p.prefill_calls > 0) has_prefill = true;
            if (p.decode_calls  > 0) has_decode  = true;
            if (p.forward_calls > 0) has_forward = true;
        }

        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                     Per-Layer Benchmark Report                          ║\n");
        printf("╚══════════════════════════════════════════════════════════════════════════╝\n");

        if (has_prefill) printPhase("Prefill", PhaseKind::Prefill);
        if (has_decode)  printPhase("Decode",  PhaseKind::Decode);
        if (has_forward) printPhase("Forward", PhaseKind::Forward);

        if (!has_prefill && !has_decode && !has_forward)
        {
            printf("  (no profiling data collected)\n\n");
        }
    }

private:
    enum class PhaseKind { Prefill, Decode, Forward };

    void printPhase(const char* title, PhaseKind kind) const
    {
        // Find max layer name width
        int max_name_len = 5; // "Layer"
        for (const auto& p : profiles_)
        {
            int len = (int)p.layerName.size();
            if (len > max_name_len) max_name_len = len;
        }
        max_name_len = std::min(max_name_len, 30);

        // Compute total for percentage
        double total_us = 0.0;
        for (const auto& p : profiles_)
        {
            double t = (kind == PhaseKind::Prefill)  ? p.prefill_total_us :
                       (kind == PhaseKind::Decode)   ? p.decode_total_us  :
                                                       p.forward_total_us;
            total_us += t;
        }

        printf("\n  ── %s ──\n", title);
        printf("  %-*s │ %6s │ %11s │ %10s │ %6s\n",
               max_name_len, "Layer", "Calls", "Total(ms)", "Avg(ms)", "%");
        // separator
        printf("  ");
        for (int i = 0; i < max_name_len; i++) printf("─");
        printf("─┼────────┼─────────────┼────────────┼────────\n");

        for (const auto& p : profiles_)
        {
            double t_us = 0.0;
            int calls = 0;
            if (kind == PhaseKind::Prefill)  { t_us = p.prefill_total_us; calls = p.prefill_calls; }
            else if (kind == PhaseKind::Decode) { t_us = p.decode_total_us; calls = p.decode_calls; }
            else                             { t_us = p.forward_total_us; calls = p.forward_calls; }

            if (calls == 0) continue;

            double t_ms  = t_us / 1000.0;
            double avg   = t_ms / calls;
            double pct   = (total_us > 0.0) ? (t_us / total_us * 100.0) : 0.0;

            printf("  %-*s │ %6d │ %9.3f   │ %8.3f   │ %5.1f%%\n",
                   max_name_len, p.layerName.c_str(), calls, t_ms, avg, pct);
        }

        double total_ms = total_us / 1000.0;
        printf("  ");
        for (int i = 0; i < max_name_len; i++) printf("─");
        printf("─┼────────┼─────────────┼────────────┼────────\n");
        printf("  %-*s │        │ %9.3f   │            │ 100.0%%\n",
               max_name_len, "TOTAL", total_ms);
        printf("\n");
    }

    std::vector<LayerProfile> profiles_;
};

} // namespace minfer

#endif // MINFER_BENCHMARK_PROFILER_H
