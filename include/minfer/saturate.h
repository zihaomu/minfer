//
// Created by mzh on 2024/8/5.
//

#ifndef MINFER_SATURATE_H
#define MINFER_SATURATE_H

#include "mat.h"
#include <cmath>
#include <cstdint>
#include <climits>

namespace minfer
{

//! @addtogroup core_utils
//! @{

/////////////// saturate_cast (used in image & signal processing) ///////////////////

/** @brief Template function for accurate conversion from one primitive type to another.

 The function saturate_cast resembles the standard C++ cast operations, such as static_cast\<T\>()
 and others. It perform an efficient and accurate conversion from one primitive type to another
 (see the introduction chapter). saturate in the name means that when the input value v is out of the
 range of the target type, the result is not formed just by taking low bits of the input, but instead
 the value is clipped. For example:
 @code
 uchar a = saturate_cast<uchar>(-100); // a = 0 (UCHAR_MIN)
 short b = saturate_cast<short>(33333.33333); // b = 32767 (SHRT_MAX)
 @endcode
 Such clipping is done when the target type is unsigned char , signed char , unsigned short or
 signed short . For 32-bit integers, no clipping is done.

 When the parameter is a floating-point value and the target type is an integer (8-, 16- or 32-bit),
 the floating-point value is first rounded to the nearest integer and then clipped if needed (when
 the target type is 8- or 16-bit).

 @param v Function parameter.
 @sa add, subtract, multiply, divide, Mat::convertTo
 */
template<typename _Tp> static inline _Tp saturate_cast(uchar v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(schar v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(ushort v)   { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(short v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(unsigned v) { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(int v)      { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(float v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(double v)   { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(int64 v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(uint64 v)   { return _Tp(v); }

template<> inline uchar saturate_cast<uchar>(schar v)        { return (uchar)std::max((int)v, 0); }
template<> inline uchar saturate_cast<uchar>(ushort v)       { return (uchar)std::min((unsigned)v, (unsigned)UCHAR_MAX); }
template<> inline uchar saturate_cast<uchar>(int v)          { return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline uchar saturate_cast<uchar>(short v)        { return saturate_cast<uchar>((int)v); }
template<> inline uchar saturate_cast<uchar>(unsigned v)     { return (uchar)std::min(v, (unsigned)UCHAR_MAX); }
template<> inline uchar saturate_cast<uchar>(float v)        { int iv = std::round(v); return saturate_cast<uchar>(iv); }
template<> inline uchar saturate_cast<uchar>(double v)       { int iv = std::round(v); return saturate_cast<uchar>(iv); }
template<> inline uchar saturate_cast<uchar>(int64 v)        { return (uchar)((uint64)v <= (uint64)UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline uchar saturate_cast<uchar>(uint64 v)       { return (uchar)std::min(v, (uint64)UCHAR_MAX); }

template<> inline schar saturate_cast<schar>(uchar v)        { return (schar)std::min((int)v, SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(ushort v)       { return (schar)std::min((unsigned)v, (unsigned)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(int v)          { return (schar)((unsigned)(v-SCHAR_MIN) <= (unsigned)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline schar saturate_cast<schar>(short v)        { return saturate_cast<schar>((int)v); }
template<> inline schar saturate_cast<schar>(unsigned v)     { return (schar)std::min(v, (unsigned)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(float v)        { int iv = std::round(v); return saturate_cast<schar>(iv); }
template<> inline schar saturate_cast<schar>(double v)       { int iv = std::round(v); return saturate_cast<schar>(iv); }
template<> inline schar saturate_cast<schar>(int64 v)        { return (schar)((uint64)((int64)v-SCHAR_MIN) <= (uint64)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline schar saturate_cast<schar>(uint64 v)       { return (schar)std::min(v, (uint64)SCHAR_MAX); }

template<> inline ushort saturate_cast<ushort>(schar v)      { return (ushort)std::max((int)v, 0); }
template<> inline ushort saturate_cast<ushort>(short v)      { return (ushort)std::max((int)v, 0); }
template<> inline ushort saturate_cast<ushort>(int v)        { return (ushort)((unsigned)v <= (unsigned)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline ushort saturate_cast<ushort>(unsigned v)   { return (ushort)std::min(v, (unsigned)USHRT_MAX); }
template<> inline ushort saturate_cast<ushort>(float v)      { int iv = std::round(v); return saturate_cast<ushort>(iv); }
template<> inline ushort saturate_cast<ushort>(double v)     { int iv = std::round(v); return saturate_cast<ushort>(iv); }
template<> inline ushort saturate_cast<ushort>(int64 v)      { return (ushort)((uint64)v <= (uint64)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline ushort saturate_cast<ushort>(uint64 v)     { return (ushort)std::min(v, (uint64)USHRT_MAX); }

template<> inline short saturate_cast<short>(ushort v)       { return (short)std::min((int)v, SHRT_MAX); }
template<> inline short saturate_cast<short>(int v)          { return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline short saturate_cast<short>(unsigned v)     { return (short)std::min(v, (unsigned)SHRT_MAX); }
template<> inline short saturate_cast<short>(float v)        { int iv = std::round(v); return saturate_cast<short>(iv); }
template<> inline short saturate_cast<short>(double v)       { int iv = std::round(v); return saturate_cast<short>(iv); }
template<> inline short saturate_cast<short>(int64 v)        { return (short)((uint64)((int64)v - SHRT_MIN) <= (uint64)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline short saturate_cast<short>(uint64 v)       { return (short)std::min(v, (uint64)SHRT_MAX); }

template<> inline int saturate_cast<int>(unsigned v)         { return (int)std::min(v, (unsigned)INT_MAX); }
template<> inline int saturate_cast<int>(int64 v)            { return (int)((uint64)(v - INT_MIN) <= (uint64)UINT_MAX ? v : v > 0 ? INT_MAX : INT_MIN); }
template<> inline int saturate_cast<int>(uint64 v)           { return (int)std::min(v, (uint64)INT_MAX); }
template<> inline int saturate_cast<int>(float v)            { return std::round(v); }
template<> inline int saturate_cast<int>(double v)           { return std::round(v); }

template<> inline unsigned saturate_cast<unsigned>(schar v)  { return (unsigned)std::max(v, (schar)0); }
template<> inline unsigned saturate_cast<unsigned>(short v)  { return (unsigned)std::max(v, (short)0); }
template<> inline unsigned saturate_cast<unsigned>(int v)    { return (unsigned)std::max(v, (int)0); }
template<> inline unsigned saturate_cast<unsigned>(int64 v)  { return (unsigned)((uint64)v <= (uint64)UINT_MAX ? v : v > 0 ? UINT_MAX : 0); }
template<> inline unsigned saturate_cast<unsigned>(uint64 v) { return (unsigned)std::min(v, (uint64)UINT_MAX); }
// we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
template<> inline unsigned saturate_cast<unsigned>(float v)  { return static_cast<unsigned>(std::round(v)); }
template<> inline unsigned saturate_cast<unsigned>(double v) { return static_cast<unsigned>(std::round(v)); }

template<> inline uint64 saturate_cast<uint64>(schar v)      { return (uint64)std::max(v, (schar)0); }
template<> inline uint64 saturate_cast<uint64>(short v)      { return (uint64)std::max(v, (short)0); }
template<> inline uint64 saturate_cast<uint64>(int v)        { return (uint64)std::max(v, (int)0); }
template<> inline uint64 saturate_cast<uint64>(int64 v)      { return (uint64)std::max(v, (int64)0); }

template<> inline int64 saturate_cast<int64>(uint64 v)       { return (int64)std::min(v, (uint64)LLONG_MAX); }

/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(hfloat v) { return saturate_cast<_Tp>((float)v); }

// in theory, we could use a LUT for 8u/8s->16f conversion,
// but with hardware support for FP32->FP16 conversion the current approach is preferable
template<> inline hfloat saturate_cast<hfloat>(uchar v)   { return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(schar v)   { return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(ushort v)  { return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(short v)   { return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(unsigned v){ return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(int v)     { return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(uint64 v)  { return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(int64 v)   { return hfloat((float)v); }
template<> inline hfloat saturate_cast<hfloat>(float v)   { return hfloat(v); }
template<> inline hfloat saturate_cast<hfloat>(double v)  { return hfloat((float)v); }
}

#endif //MINFER_SATURATE_H
