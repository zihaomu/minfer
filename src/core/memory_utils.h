//
// Created by mzh on 2023/11/2.
//

#ifndef MINFER_MEMORY_UTILS_H
#define MINFER_MEMORY_UTILS_H

#include <stdio.h>
#include "minfer/define.h"
#include "define.impl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define M_MEMORY_ALIGN_DEFAULT 64 // 通用内存对齐到 64 字节

/**
 * @brief alloc memory with given size & alignment.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
M_PUBLIC void* MMemoryAllocAlign(size_t size, size_t align);

/**
 * @brief alloc memory with given size & alignment, and fill memory space with 0.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
M_PUBLIC void* MMemoryCallocAlign(size_t size, size_t align);

/**
 * @brief free aligned memory pointer.
 * @param mem   aligned memory pointer.
 * @warning do NOT pass any pointer NOT returned by `MNNMemoryAllocAlign` or `MNNMemoryCallocAlign`.
 * @sa MNNMemoryAllocAlign
 * @sa MNNMemoryCallocAlign
 */
M_PUBLIC void MMemoryFreeAlign(void* mem);


#ifdef __cplusplus
}
#endif
#endif //MINFER_MEMORY_UTILS_H
