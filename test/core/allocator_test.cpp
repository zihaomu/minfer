//
// Created by mzh on 2023/11/2.
//

#include "../../src/core/allocator.h"
#include "gtest/gtest.h"

using namespace minfer;

TEST(Allocator, memory_test)
{
    std::shared_ptr<Allocator> allocator;
    allocator.reset(new Allocator(Allocator::AllocatorImpl::createDefault()));

    float* tp1 = (float *)malloc(1000 );
    float* tp2 = (float *)malloc(1000 );
    float* tp3 = (float *)malloc(1000 );

    // allocator memory 1
    auto p1 = allocator->alloc(1000);

    // allocate memory 2
    auto p2 = allocator->alloc(2000);

    // allocate memory 3
    auto p3 = allocator->alloc(3000);

    // free memory 1
    allocator->free(p1);

    // free memory 2
    allocator->free(p2);

    // allocate memory 4, it should resue the memory 1.
    auto p4 = allocator->alloc(1100);

    M_Assert(p2.first == p4.first);
}