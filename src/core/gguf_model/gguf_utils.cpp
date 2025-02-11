//
// Created by mzh on 2024/7/23.
//

#include <stdint.h>
#include <inttypes.h>
#include "gguf_utils.h"
#include "ggml_quant.h"

#ifdef _WIN32
#include <windows.h>
#endif
namespace minfer {

GGUF_context::~GGUF_context() {
    if (this->kv) {
        // free string memory - not great..
        for (uint64_t i = 0; i < this->header.n_kv; ++i) {
            struct GGUF_kv *kv = &this->kv[i];

            if (kv->key.data) {
                MMemoryFreeAlign(kv->key.data);
            }

            if (kv->type == GGUF_TYPE_STRING) {
                if (kv->value.str.data) {
                    MMemoryFreeAlign(kv->value.str.data);
                }
            }

            if (kv->type == GGUF_TYPE_ARRAY) {
                if (kv->value.arr.data) {
                    if (kv->value.arr.type == GGUF_TYPE_STRING) {
                        for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                            struct GGUF_str *str = &((struct GGUF_str *) kv->value.arr.data)[j];
                            if (str->data) {
                                MMemoryFreeAlign(str->data);
                            }
                        }
                    }
                    MMemoryFreeAlign(kv->value.arr.data);
                }
            }
        }

        MMemoryFreeAlign(this->kv);
    }

    if (this->tensors) {
        for (uint64_t i = 0; i < this->header.n_tensors; ++i) {
            struct GGUF_tensor *tensors = &this->tensors[i];

            if (tensors->name.data) {
                MMemoryFreeAlign(tensors->name.data);
            }
        }

        MMemoryFreeAlign(this->tensors);
    }

    if (this->data) {
        MMemoryFreeAlign(this->data);
    }
}

#ifdef _WIN32
    static wchar_t * ggml_mbstowcs(const char * mbs) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, NULL, 0);
        if (!wlen) {
            errno = EINVAL;
            return NULL;
        }

        wchar_t * wbuf = (wchar_t *)MMemoryAllocAlign(wlen * sizeof(wchar_t));
        wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
        if (!wlen) {
            MMemoryFreeAlign(wbuf);
            errno = EINVAL;
            return NULL;
        }

        return wbuf;
    }
#endif

FILE *gguf_fopen(const char *fname, const char *mode) {
#ifdef _WIN32
    FILE * file = NULL;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    if (wfname) {
        // convert mode (ANSI)
        wchar_t * wmode = (wchar_t *)MMemoryAllocAlign((strlen(mode) + 1) * sizeof(wchar_t));
        wchar_t * wmode_p = wmode;
        do {
            *wmode_p++ = (wchar_t)*mode;
        } while (*mode++);

        // open file
        file = _wfopen(wfname, wmode);

        MMemoryFreeAlign(wfname);
        MMemoryFreeAlign(wmode);
    }

    return file;
#else
    return fopen(fname, mode);
#endif
}

// read element with given offset and given length.
bool gguf_fread_el(FILE *file, void *dst, size_t size, size_t *offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

bool gguf_fread_str(FILE *file, GGUF_str *p, size_t *offset) {
    p->n = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        M_ERROR("alloc memory error %s: invalid string length %d!", __func__, (int) p->n);
        return false;
    }

    p->data = (char *) MMemoryCallocAlign(p->n + 1, M_MEMORY_ALIGN_DEFAULT);

    ok = ok && gguf_fread_el(file, p->data, p->n, offset);

    return ok;
}

int gguf_get_n_tensors(const struct GGUF_context *ctx) {
    return ctx->header.n_tensors;
}

char *gguf_get_tensor_name(const struct GGUF_context *ctx, int i) {
    return ctx->tensors[i].name.data;
}

int gguf_find_tensor(const struct GGUF_context *ctx, const char *name) {
    // return -1 if tensor not found
    int tensorfound = -1;

    const int n_tensors = gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        if (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensorfound = i;
            break;
        }
    }

    return tensorfound;
}

// get block size, based on type.
int ggml_blck_size(enum GGML_TYPE type) {
    return typeTraits[type].blck_size;
}

size_t ggml_type_size(enum GGML_TYPE type) {
    return typeTraits[type].type_size;
}

const char *ggml_type_name(enum GGML_TYPE type) {
    return typeTraits[type].type_name;
}

size_t ggml_row_size(enum GGML_TYPE type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type) * ne / ggml_blck_size(type);
}

size_t gguf_type_size(enum GGUF_TYPE type) {
    M_Assert(0 <= type && type < GGUF_TYPE_COUNT);
    auto it = GGUF_TYPE_SIZE.find(type);
    return it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

const char * GGUF_TYPE_name(enum GGUF_TYPE type) {
    auto it = GGUF_TYPE_NAME.find(type);
    return it == GGUF_TYPE_NAME.end() ? 0 : it->second;
}

int gguf_get_version(const struct GGUF_context * ctx) {
    return ctx->header.version;
}

size_t gguf_get_alignment(const struct GGUF_context * ctx) {
    return ctx->alignment;
}

size_t gguf_get_data_offset(const struct GGUF_context * ctx) {
    return ctx->offset;
}

void * gguf_get_data(const struct GGUF_context * ctx) {
    return ctx->data;
}

int gguf_get_n_kv(const struct GGUF_context * ctx) {
    return ctx->header.n_kv;
}

int gguf_find_key(const struct GGUF_context * ctx, const char * key) {
    // return -1 if key not found
    int keyfound = -1;

    const int n_kv = gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

//>>>>>>>>>>>>>>>>>>>>>> get basic data BEGIN <<<<<<<<<<<<<<<<<<<<<<<<<<<<
const char * gguf_get_key(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].key.data;
}

enum GGUF_TYPE gguf_get_kv_type(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].type;
}

enum GGUF_TYPE gguf_get_arr_type(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.type;
}

const void * gguf_get_arr_data(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.data;
}

const char * gguf_get_arr_str(const struct GGUF_context * ctx, int key_id, int i) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    struct GGUF_kv * kv = &ctx->kv[key_id];
    struct GGUF_str * str = &((struct GGUF_str *) kv->value.arr.data)[i];
    return str->data;
}

int gguf_get_arr_n(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.n;
}

uint8_t gguf_get_val_u8(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_UINT8);
    return ctx->kv[key_id].value.uint8;
}

int8_t gguf_get_val_i8(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_INT8);
    return ctx->kv[key_id].value.int8;
}

uint16_t gguf_get_val_u16(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_UINT16);
    return ctx->kv[key_id].value.uint16;
}

int16_t gguf_get_val_i16(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_INT16);
    return ctx->kv[key_id].value.int16;
}

uint32_t gguf_get_val_u32(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}

int32_t gguf_get_val_i32(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_INT32);
    return ctx->kv[key_id].value.int32;
}

float gguf_get_val_f32(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_FLOAT32);
    return ctx->kv[key_id].value.float32;
}

uint64_t gguf_get_val_u64(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_UINT64);
    return ctx->kv[key_id].value.uint64;
}

int64_t gguf_get_val_i64(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_INT64);
    return ctx->kv[key_id].value.int64;
}

double gguf_get_val_f64(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_FLOAT64);
    return ctx->kv[key_id].value.float64;
}

bool gguf_get_val_bool(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_BOOL);
    return ctx->kv[key_id].value.bool_;
}

const char * gguf_get_val_str(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type == GGUF_TYPE_STRING);
    return ctx->kv[key_id].value.str.data;
}

const void * gguf_get_val_data(const struct GGUF_context * ctx, int key_id) {
    M_Assert(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_Assert(ctx->kv[key_id].type != GGUF_TYPE_ARRAY);
    M_Assert(ctx->kv[key_id].type != GGUF_TYPE_STRING);
    return &ctx->kv[key_id].value;
}

//>>>>>>>>>>>>>>>>>>>>>> get basic data END <<<<<<<<<<<<<<<<<<<<<<<<<<<<

LLM_ARCH llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

}