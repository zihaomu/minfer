//
// Created by Codex on 2026/4/11.
//

#include "benchmark_profiler.h"
#include "gtest/gtest.h"

using namespace minfer;

TEST(BenchmarkProfiler_TEST, report_includes_decode_substages)
{
    BenchmarkProfiler profiler;
    profiler.resize(1);
    profiler.setLayerInfo(0, "LmHeadLayer_19", LayerType::LmHead);

    profiler.record(0, InferPhase::Decode, 1000.0);
    profiler.recordSubEvent(0, InferPhase::Decode, "selectNT", 800.0);
    profiler.recordSubEvent(0, InferPhase::Decode, "dispatch", 100.0);

    const std::string report = profiler.reportString();
    EXPECT_NE(report.find("Decode Sub-Stages"), std::string::npos);
    EXPECT_NE(report.find("LmHeadLayer_19/selectNT"), std::string::npos);
    EXPECT_NE(report.find("LmHeadLayer_19/dispatch"), std::string::npos);
}
