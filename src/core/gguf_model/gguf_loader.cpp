//
// Created by mzh on 2024/4/10.
//

#include <algorithm>
#include <iostream>
#include <cstring>
#include <memory>
#include <map>
#include <iostream>
#include <fstream>
#include <stdint.h>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#include <fcntl.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#include "gguf_loader.h"

#include "../memory_utils.h"
#include "ggml.h"

namespace minfer
{
#define GGML_MAX_NAME           64
#define GGML_MAX_DIMS           4
#define GGUF_DEFAULT_ALIGNMENT 32
//>>>>>>>>>>>>>>>>>>>>>> common  <<<<<<<<<<<<<<<<<<<<<<<<<<<<

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    M_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    M_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

#define GGUF_MAGIC "GGUF"

//>>>>>>>>>>>>>>>>>>>>>> GGUF common  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
struct GGUF_header
{
    char magic[4];       // GGUF

    uint32_t version;
    uint64_t n_tensors; // kv tensor number, GGUV V2
    uint64_t n_kv;      // kv number
};

struct GGUF_str {
    uint64_t n; // length of string, GGUFv2
    char* data; // point of data.
};

static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
        [GGUF_TYPE_UINT8]   = sizeof(uint8_t),
        [GGUF_TYPE_INT8]    = sizeof(int8_t),
        [GGUF_TYPE_UINT16]  = sizeof(uint16_t),
        [GGUF_TYPE_INT16]   = sizeof(int16_t),
        [GGUF_TYPE_UINT32]  = sizeof(uint32_t),
        [GGUF_TYPE_INT32]   = sizeof(int32_t),
        [GGUF_TYPE_FLOAT32] = sizeof(float),
        [GGUF_TYPE_BOOL]    = sizeof(bool),
        [GGUF_TYPE_STRING]  = sizeof(struct GGUF_str),
        [GGUF_TYPE_UINT64]  = sizeof(uint64_t),
        [GGUF_TYPE_INT64]   = sizeof(int64_t),
        [GGUF_TYPE_FLOAT64] = sizeof(double),
        [GGUF_TYPE_ARRAY]   = 0, // undefined
};

static const char * GGUF_TYPE_NAME[GGUF_TYPE_COUNT] = {
        [GGUF_TYPE_UINT8]   = "u8",
        [GGUF_TYPE_INT8]    = "i8",
        [GGUF_TYPE_UINT16]  = "u16",
        [GGUF_TYPE_INT16]   = "i16",
        [GGUF_TYPE_UINT32]  = "u32",
        [GGUF_TYPE_INT32]   = "i32",
        [GGUF_TYPE_FLOAT32] = "f32",
        [GGUF_TYPE_BOOL]    = "bool",
        [GGUF_TYPE_STRING]  = "str",
        [GGUF_TYPE_ARRAY]   = "arr",
        [GGUF_TYPE_UINT64]  = "u64",
        [GGUF_TYPE_INT64]   = "i64",
        [GGUF_TYPE_FLOAT64] = "f64",
};

static size_t gguf_type_size(enum GGUF_TYPE type) {
    M_ASSERT(0 <= type && type < GGUF_TYPE_COUNT);
    return GGUF_TYPE_SIZE[type];
}

union GGUF_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct GGUF_str str; // string data type,

    struct { // arrray data type
        GGUF_TYPE type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct GGUF_kv
{
    struct GGUF_str key;
    GGUF_TYPE type;
    GGUF_value value;
};

// TODO how to be compatible with the V1 and V2 version
struct GGUF_tensor_info {
    struct GGUF_str name;

    uint32_t n_dims;
    uint64_t  ne[GGML_MAX_NAME];

    GGML_TYPE type;

    uint64_t offset;

    const void* data;
    size_t size;
};

// This file keep all the gguf model.
struct GGUF_context {
    struct GGUF_header header;
    struct GGUF_kv *kv;              // pointer to all the kv list. What is kv? kv means the gguf key-value data struct, and the kv list contains all key-value info where the model has.
    struct GGUF_tensor_info* infos;  // pointer to all tensor info list.

    size_t alignment;
    size_t offset;     // offset of data from beginning of file.
    size_t size;       // size of data in bytes

    void* data;

~GGUF_context();
};

GGUF_context::~GGUF_context()
{
    if (this->kv) {
        // free string memory - not great..
        for (uint64_t i = 0; i < this->header.n_kv; ++i) {
            struct GGUF_kv * kv = &this->kv[i];

            if (kv->key.data) {
                MMemoryFreeAlign(kv->key.data);
            }

            if (kv->type == GGUF_TYPE_STRING) {
                if (kv->value.str.data) {
                    MMemoryFreeAlign(kv->value.str.data);
                }
            }

            if (kv->type == GGUF_TYPE_ARRAY)
            {
                if (kv->value.arr.data)
                {
                    if (kv->value.arr.type == GGUF_TYPE_STRING)
                    {
                        for (uint64_t j = 0; j < kv->value.arr.n; ++j)
                        {
                            struct GGUF_str * str = &((struct GGUF_str *) kv->value.arr.data)[j];
                            if (str->data)
                            {
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

    if (this->infos)
    {
        for (uint64_t i = 0; i < this->header.n_tensors; ++i)
        {
            struct GGUF_tensor_info * info = &this->infos[i];

            if (info->name.data)
            {
                MMemoryFreeAlign(info->name.data);
            }
        }

        MMemoryFreeAlign(this->infos);
    }
}

//>>>>>>>>>>>>>>>>>>>>>> LLama common  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
FILE* gguf_fopen(const char* fname, const char* mode)
{
#ifdef _WIN32
#error "No implement!"
#else
    return fopen(fname, mode);
#endif
}

// read element with given offset and given length.
static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE* file, GGUF_str* p, size_t* offset)
{
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        M_ERROR("alloc memory error %s: invalid string length %d!", __func__, (int)p->n);
        return false;
    }

    p->data = (char *)MMemoryCallocAlign(p->n + 1, M_MEMORY_ALIGN_DEFAULT);

    ok = ok && gguf_fread_el(file,  p->data, p->n, offset);

    return ok;
}

struct LLama_file {
    FILE* fp;
    size_t size;

    LLama_file(const char* fname, const char* mode)
    {
        fp = gguf_fopen(fname, mode);

        if (fp == NULL)
            throw std::runtime_error(format("failed to open: %s : %s", fname, strerror(errno)));

        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET); // moving pointer to set offset
    }

    // return pointer moved length.
    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        M_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    // move pointer
    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        M_ASSERT(ret == 0); // same
    }

    // read raw
    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~LLama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

struct LLama_mmap {
    void* addr;
    size_t size;

    LLama_mmap(const LLama_mmap&) = delete;

    std::vector<std::pair<size_t, size_t > > mapped_fragments;
    LLama_mmap(struct LLama_file *file, size_t prefetch = (size_t ) - 1)
    {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = MAP_SHARED;

        addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);

        if (addr == MAP_FAILED)
        {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0)
        {
            // advise the kernel to preload the mapped memory
            if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED))
            {
                std::cout<<"warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed!"<<std::endl;
            }
        }

        mapped_fragments.emplace_back(0, file->size);
    }

    static void align_range(size_t* first, size_t* last, size_t page_size)
    {
        // align first to the next page
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        // align last to the previous page
        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    ~LLama_mmap()
    {
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                std::cout<<"warning: munmap failed!"<<std::endl;
            }
        }
    }
};

struct LLama_context {
    struct GGUF_header *header;
    struct GGUF_kv *kv;
    struct GGUF_tensor_info *info;

    size_t alignment;
    size_t offset;
    size_t size;

    void* data;
};

enum LLama_model_kv_override_type
{
    LLAMA_KV_OVERRIDE_TYPE_INT,
    LLAMA_KV_OVERRIDE_TYPE_FLOAT,
    LLAMA_KV_OVERRIDE_TYPE_BOOL,
};

struct LLama_model_kv_override
{
    char key[128];
    LLama_model_kv_override_type tag;

    // TODO Why the float is double, and the int64_t is int.
    union {
        int64_t int_value;
        double float_value;
        bool bool_value;
    };
};

// sanity check
static void gguf_tensor_info_sanitize(struct GGUF_tensor_info* info)
{
    M_ASSERT(info->n_dims <= GGML_MAX_DIMS);
    M_ASSERT(0 <= info->type && info->type < GGML_TYPE_COUNT);

    for (uint32_t i = 0; i < info->n_dims; ++i)
    {
        M_ASSERT(info->ne[i] > 0);
    }

    // prevent overflow for total number of elements.
    M_ASSERT(INT64_MAX/info->ne[1] > info->ne[0]);
    M_ASSERT(INT64_MAX/info->ne[2] > info->ne[0]*info->ne[1]);
    M_ASSERT(INT64_MAX/info->ne[3] > info->ne[0]*info->ne[1]*info->ne[2]);
}

void gguf_free(struct GGUF_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->kv) {
        // free string memory - not great..
        for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
            struct GGUF_kv * kv = &ctx->kv[i];

            if (kv->key.data) {
                MMemoryFreeAlign(kv->key.data);
            }

            if (kv->type == GGUF_TYPE_STRING) {
                if (kv->value.str.data) {
                    MMemoryFreeAlign(kv->value.str.data);
                }
            }

            if (kv->type == GGUF_TYPE_ARRAY)
            {
                if (kv->value.arr.data)
                {
                    if (kv->value.arr.type == GGUF_TYPE_STRING)
                    {
                        for (uint64_t j = 0; j < kv->value.arr.n; ++j)
                        {
                            struct GGUF_str * str = &((struct GGUF_str *) kv->value.arr.data)[j];
                            if (str->data)
                            {
                                MMemoryFreeAlign(str->data);
                            }
                        }
                    }
                    MMemoryFreeAlign(kv->value.arr.data);
                }
            }
        }

        MMemoryFreeAlign(ctx->kv);
    }

    if (ctx->infos) {
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct GGUF_tensor_info * info = &ctx->infos[i];

            if (info->name.data) {
                MMemoryFreeAlign(info->name.data);
            }
        }

        MMemoryFreeAlign(ctx->infos);
    }

    MMemoryFreeAlign(ctx);
}

const char * gguf_type_name(enum GGUF_TYPE type) {
    return GGUF_TYPE_NAME[type];
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

const char * gguf_get_key(const struct GGUF_context * ctx, int key_id) {
    M_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].key.data;
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

uint32_t gguf_get_val_u32(const struct GGUF_context * ctx, int key_id) {
    M_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    M_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}

std::shared_ptr<GGUF_context> gguf_init_from_file(const char* fname)
{
    std::shared_ptr<GGUF_context> ctx = std::make_shared<GGUF_context>();

    // load function
    {
        FILE * file = gguf_fopen(fname, "rb");

        if (!file)
        {
            M_ERROR("Error to load file %s!\n", fname);
            return nullptr;
        }

        size_t offset = 0;

        char magic[4];

        // check the magic
        {
            gguf_fread_el(file, &magic, sizeof (magic), &offset);

            for (uint32_t i = 0; i < sizeof(magic); i++)
            {
                if (magic[i] != GGUF_MAGIC[i])
                {
                    M_ERROR("%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                    fclose(file);
                    return nullptr;
                }
            }
        }

        bool ok = true;

        // read header
        {
            strncpy(ctx->header.magic, magic, 4);

            ctx->kv    = nullptr;
            ctx->infos = nullptr;
            ctx->data  = nullptr;

            ok = ok & gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
            ok = ok & gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
            ok = ok & gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

            // Add code is compatible with version 1.
            if (ctx->header.version == 1)
            {
                M_ERROR("%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
                fclose(file);
                return nullptr;
            }

            // sanity-checks to prevent from integer/buffer overflows.
            ok = ok & (ctx->header.n_tensors < (SIZE_MAX/2)/sizeof(GGUF_tensor_info));
            ok = ok & (ctx->header.n_kv      < (SIZE_MAX/2)/sizeof(GGUF_kv));

            if (!ok)
            {
                fprintf(stderr, "%s: failed to read header\n", __func__);
                fclose(file);
                ctx.reset();
                return nullptr;
            }
        }

        // read kv pairs
        {
            ctx->kv = (GGUF_kv *)MMemoryAllocAlign(ctx->header.n_kv * sizeof(struct GGUF_kv), M_MEMORY_ALIGN_DEFAULT);

            for (uint64_t i = 0; i < ctx->header.n_kv; ++i)
            {
                GGUF_kv* kv = &ctx->kv[i];

                ok = ok && gguf_fread_str(file, &kv->key, &offset);                       // read kv-key
                ok = ok && gguf_fread_el(file, &kv->type, sizeof(kv->type), &offset);     // read kv-type

                switch (kv->type) {
                    // read single value type, only need to load the value to the specific data type.
                    case GGUF_TYPE_UINT8:   ok = ok && gguf_fread_el (file, &kv->value.uint8,   sizeof(kv->value.uint8),   &offset); break;
                    case GGUF_TYPE_INT8:    ok = ok && gguf_fread_el (file, &kv->value.int8,    sizeof(kv->value.int8),    &offset); break;
                    case GGUF_TYPE_UINT16:  ok = ok && gguf_fread_el (file, &kv->value.uint16,  sizeof(kv->value.uint16),  &offset); break;
                    case GGUF_TYPE_INT16:   ok = ok && gguf_fread_el (file, &kv->value.int16,   sizeof(kv->value.int16),   &offset); break;
                    case GGUF_TYPE_UINT32:  ok = ok && gguf_fread_el (file, &kv->value.uint32,  sizeof(kv->value.uint32),  &offset); break;
                    case GGUF_TYPE_INT32:   ok = ok && gguf_fread_el (file, &kv->value.int32,   sizeof(kv->value.int32),   &offset); break;
                    case GGUF_TYPE_FLOAT32: ok = ok && gguf_fread_el (file, &kv->value.float32, sizeof(kv->value.float32), &offset); break;
                    case GGUF_TYPE_UINT64:  ok = ok && gguf_fread_el (file, &kv->value.uint64,  sizeof(kv->value.uint64),  &offset); break;
                    case GGUF_TYPE_INT64:   ok = ok && gguf_fread_el (file, &kv->value.int64,   sizeof(kv->value.int64),   &offset); break;
                    case GGUF_TYPE_FLOAT64: ok = ok && gguf_fread_el (file, &kv->value.float64, sizeof(kv->value.float64), &offset); break;
                    case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
                    case GGUF_TYPE_STRING:  ok = ok && gguf_fread_str(file, &kv->value.str,                                &offset); break;
                    case GGUF_TYPE_ARRAY:
                    {
                        // get array type and length.
                        ok = ok && gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);

                        switch (kv->value.arr.type) {
                            case GGUF_TYPE_UINT8:
                            case GGUF_TYPE_INT8:
                            case GGUF_TYPE_UINT16:
                            case GGUF_TYPE_INT16:
                            case GGUF_TYPE_UINT32:
                            case GGUF_TYPE_INT32:
                            case GGUF_TYPE_FLOAT32:
                            case GGUF_TYPE_UINT64:
                            case GGUF_TYPE_INT64:
                            case GGUF_TYPE_FLOAT64:
                            case GGUF_TYPE_BOOL:
                            {
                                // prevent from integer overflow in the malloc below
                                if (kv->value.arr.n >= SIZE_MAX/gguf_type_size(kv->value.arr.type)) {
                                    M_ERROR("%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
                                    fclose(file);
                                    return NULL;
                                }

                                kv->value.arr.data = MMemoryAllocAlign(kv->value.arr.n * gguf_type_size(kv->value.arr.type));

                                ok = ok && gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * gguf_type_size(kv->value.arr.type), &offset);
                            } break;
                            case GGUF_TYPE_STRING:  // for string array, used to save tokens or vocabularies. The length of each string varies.
                            {
                                // prevent from integer overflow in the malloc below
                                if (kv->value.arr.n >= SIZE_MAX/sizeof(struct GGUF_str)) {
                                    M_ERROR("%s: array size is too large %d ! \n", __func__, (int)kv->value.arr.n);
                                    fclose(file);
                                    return NULL;
                                }

                                kv->value.arr.data = MMemoryAllocAlign(kv->value.arr.n * sizeof(struct GGUF_str));

                                // iterate through all string array
                                for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                                    // allocate memory inside the func.
                                    ok = ok && gguf_fread_str(file, &((struct GGUF_str *) kv->value.arr.data)[j], &offset);
                                }
                            } break;
                            case GGUF_TYPE_ARRAY:
                            default: M_ASSERT(false && "invalid type"); break;
                        }
                    } break;
                    default: M_ASSERT(false && "invalid type");
                }

                if (!ok) {
                    break;
                }
            }

            if (!ok)
            {
                fprintf(stderr, "%s: failed to read header\n", __func__);
                fclose(file);
                ctx.reset();
                return nullptr;
            }
        }

        // read the tensor infos
        {
            ctx->infos = (GGUF_tensor_info *)MMemoryAllocAlign(ctx->header.n_tensors * sizeof (GGUF_tensor_info));

            for (uint64_t i = 0; i < ctx->header.n_tensors; i++)
            {
                GGUF_tensor_info* info = &ctx->infos[i];

                // set default dim
                for (int j = 0; j < GGML_MAX_DIMS; j++)
                {
                    info->ne[j] = 1;
                }

                ok = ok && gguf_fread_str(file, &info->name,  &offset);
                ok = ok && gguf_fread_el(file, &info->n_dims, sizeof(info->n_dims), &offset);

                ok = ok && (info->n_dims <= GGML_MAX_DIMS);

                // get right dim
                for (uint32_t j = 0; j < info->n_dims; ++j)
                {
                    ok = ok && gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
                }

                ok = ok && gguf_fread_el(file, &info->type, sizeof(info->type), &offset);
                ok = ok && gguf_fread_el(file, &info->offset, sizeof(info->offset), &offset);

                gguf_tensor_info_sanitize(info);

                if (!ok) {
                    fprintf(stderr, "%s: failed to read tensor info\n", __func__);
                    fclose(file);
                    ctx.reset();
                    return NULL;
                }
            };
        }

        ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

        int alignment_idx = gguf_find_key(ctx.get(), "general.alignment");

        if (alignment_idx != -1)
        {
            ctx->alignment = gguf_get_val_u32(ctx.get(), alignment_idx);
        }

        // take in account the padding for data alignment.
        {
            const size_t offset_pad = offset % ctx->alignment;
            if (offset_pad != 0)
            {
                offset += ctx->alignment - offset_pad;
                fseek(file, offset, SEEK_SET);
            }
        }

        // store the current file offset - this is where the data section starts.
        ctx->offset = offset;

        // compute the total size of the data section, taking into account the alignment.
        {
            ctx->size = 0;

            for (uint64_t i =0; i < ctx->header.n_tensors; ++i)
            {
                GGUF_tensor_info* info = &ctx->infos[i];

                const int64_t ne =
                        (int64_t) info->ne[0] * (int64_t) info->ne[1] *
                        (int64_t) info->ne[2] * (int64_t) info->ne[3];

                // need every data type
                if (ne % ggml_bloc)
                {

                }
            }
        }

    }
}

struct LLama_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t n_bytes = 0;

    std::unique_ptr<LLama_file> file;
    std::unique_ptr<LLama_mmap> mapping;

    struct LLama_tensor_weights {
        uint16_t idx; // source file idex
        size_t offs;  // tensor data offset in the original file.
        void *data;
        char name[GGML_MAX_NAME];

        // TODO here
        LLama_tensor_weights(uint16_t idx, const char *name)
                : idx(idx) {
            // TODO fix
        }
    };

    std::vector<LLama_tensor_weights> weights;

    // kv_overrides is used for user reset model hyper-params for fine-tuning the model effect.
    std::unordered_map<std::string, struct LLama_model_kv_override> kv_overrides;

    struct GGUF_context* meta = NULL;
//        std::vector<GGML_context* > contexts; //TODO What the context for?

    std::string arch_name;
    LLM_KV llmKv = LLM_KV(LLM_ARCH_UNKNOWN);

    // MINFER_LOG_LEVEL
    // TODO support the param overrider p argument!
    LLama_loader(const std::string& fname, bool use_mmap, const struct LLama_model_kv_override* param_overrides_p)
    {
        int trace = 0;
        M_ASSERT(param_overrides_p == nullptr && "Currently, do not support the param override!")
        if (getenv("MINFER_LOG_LEVEL"))
        {
            trace = atoi(getenv("MINFER_LOG_LEVEL"));
        }

        char split_path[PATH_MAX] = {0};
        std::shared_ptr<GGUF_context> ctx = gguf_init_from_file(fname.c_str());
    }
};

void readGGUF(const std::string path, std::map<int, std::shared_ptr<LayerParams> >& netParams)
{

}

}