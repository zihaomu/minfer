#include "transpose_kernel.h"
#include "openmp_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <xsimd/xsimd.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace {

template<typename T>
void transpose2d_tiled(const unsigned char* src_raw, unsigned char* dst_raw, int rows, int cols)
{
    const T* src = reinterpret_cast<const T*>(src_raw);
    T* dst = reinterpret_cast<T*>(dst_raw);

    constexpr int TILE = 32;
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(static_cast<size_t>((rows + TILE - 1) / TILE), static_cast<size_t>(TILE) * static_cast<size_t>(cols), 1LL << 14, 2))
#endif
    for (int row0 = 0; row0 < rows; row0 += TILE)
    {
        const int row1 = std::min(row0 + TILE, rows);
        for (int col0 = 0; col0 < cols; col0 += TILE)
        {
            const int col1 = std::min(col0 + TILE, cols);
            for (int row = row0; row < row1; ++row)
            {
                for (int col = col0; col < col1; ++col)
                {
                    dst[static_cast<size_t>(col) * rows + row] = src[static_cast<size_t>(row) * cols + col];
                }
            }
        }
    }
}

template<typename T>
void transpose2d_xsimd(const unsigned char* src_raw, unsigned char* dst_raw, int rows, int cols)
{
    const T* src = reinterpret_cast<const T*>(src_raw);
    T* dst = reinterpret_cast<T*>(dst_raw);

    using batch_type = xsimd::batch<T>;
    constexpr int N = batch_type::size;
    constexpr int TILE = 64; // Tile should be a multiple of N (which is usually 4, 8, 16)

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(static_cast<size_t>((rows + TILE - 1) / TILE), static_cast<size_t>(TILE) * static_cast<size_t>(cols), 1LL << 14, 2))
#endif
    for (int row0 = 0; row0 < rows; row0 += TILE)
    {
        const int row1 = std::min(row0 + TILE, rows);
        for (int col0 = 0; col0 < cols; col0 += TILE)
        {
            const int col1 = std::min(col0 + TILE, cols);

            int row = row0;
            for (; row + N <= row1; row += N)
            {
                int col = col0;
                for (; col + N <= col1; col += N)
                {
                    batch_type matrix[N];
                    for (int i = 0; i < N; ++i)
                    {
                        matrix[i] = batch_type::load_unaligned(src + static_cast<size_t>(row + i) * cols + col);
                    }
                    
                    xsimd::transpose(matrix, matrix + N);
                    
                    for (int i = 0; i < N; ++i)
                    {
                        matrix[i].store_unaligned(dst + static_cast<size_t>(col + i) * rows + row);
                    }
                }
                // Handle remaining columns in this block of N rows
                for (; col < col1; ++col)
                {
                    for (int i = 0; i < N; ++i)
                    {
                        dst[static_cast<size_t>(col) * rows + row + i] = src[static_cast<size_t>(row + i) * cols + col];
                    }
                }
            }
            // Handle remaining rows in this TILE block
            for (; row < row1; ++row)
            {
                for (int col = col0; col < col1; ++col)
                {
                    dst[static_cast<size_t>(col) * rows + row] = src[static_cast<size_t>(row) * cols + col];
                }
            }
        }
    }
}

}  // namespace

void transpose2d_kernel_blocked(const unsigned char* src,
                                unsigned char* dst,
                                int rows,
                                int cols,
                                size_t elem_size)
{
    switch (elem_size)
    {
        case 1:
            transpose2d_tiled<uint8_t>(src, dst, rows, cols);
            break;
        case 2:
            transpose2d_xsimd<uint16_t>(src, dst, rows, cols);
            break;
        case 4:
            // uint32_t path can be architecture-sensitive in xsimd transpose;
            // float batches use the stable SIMD transpose implementation.
            transpose2d_xsimd<float>(src, dst, rows, cols);
            break;
        case 8:
            transpose2d_tiled<uint64_t>(src, dst, rows, cols);
            break;
        default:
        {
            constexpr int TILE = 32;
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(static_cast<size_t>((rows + TILE - 1) / TILE), static_cast<size_t>(TILE) * static_cast<size_t>(cols), 1LL << 14, 2))
#endif
            for (int row0 = 0; row0 < rows; row0 += TILE)
            {
                const int row1 = std::min(row0 + TILE, rows);
                for (int col0 = 0; col0 < cols; col0 += TILE)
                {
                    const int col1 = std::min(col0 + TILE, cols);
                    for (int row = row0; row < row1; ++row)
                    {
                        for (int col = col0; col < col1; ++col)
                        {
                            std::memcpy(dst + (static_cast<size_t>(col) * rows + row) * elem_size,
                                        src + (static_cast<size_t>(row) * cols + col) * elem_size,
                                        elem_size);
                        }
                    }
                }
            }
            break;
        }
    }
}

}  // namespace cpu
}  // namespace minfer
