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
#include <inttypes.h>

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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#include <io.h>
#endif

#include "gguf_loader.h"
#include "../memory_utils.h"
#include "ggml_quant.h"
#include "gguf_utils.h"
#include <queue>
#include <forward_list>

namespace minfer
{
//>>>>>>>>>>>>>>>>>>>>>> LLama common  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        M_Assert(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    // move pointer
    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        M_Assert(ret == 0); // same
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
#if defined(_WIN32)
    HANDLE fp_win32;
#endif

    LLama_mmap(const LLama_mmap&) = delete;

    std::vector<std::pair<size_t, size_t > > mapped_fragments;

    // support numa
    LLama_mmap(struct LLama_file *file, size_t prefetch = (size_t ) - 1)
    {

#ifdef _POSIX_MAPPED_FILES
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
#elif defined(_WIN32)
        size = file->size;

        HANDLE hFile = (HANDLE) _get_osfhandle(_fileno(file->fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: LLama_mmap: format win error!\n"));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: LLama_mmap: format win error! \n"));
        }

#else
        throw std::runtime_error("mmap not supported");
#endif

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
#ifdef _POSIX_MAPPED_FILES
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                std::cout<<"warning: munmap failed!"<<std::endl;
            }
        }
#elif defined(_WIN32)
        if (!UnmapViewOfFile(addr)) {
            M_Error(NULL, "warning: UnmapViewOfFile failed: llama release error in Windows!\n");
        }
#endif
    }
};

// sanity check
static void gguf_tensor_info_sanitize(struct GGUF_tensor* info)
{
    M_Assert(info->n_dims <= GGML_MAX_DIMS);
    M_Assert(0 <= info->type && info->type < GGML_TYPE_COUNT);

    for (uint32_t i = 0; i < info->n_dims; ++i)
    {
        M_Assert(info->ne[i] > 0);
    }

    // prevent overflow for total number of elements.
    M_Assert(INT64_MAX/info->ne[1] > info->ne[0]);
    M_Assert(INT64_MAX/info->ne[2] > info->ne[0]*info->ne[1]);
    M_Assert(INT64_MAX/info->ne[3] > info->ne[0]*info->ne[1]*info->ne[2]);
}

std::shared_ptr<GGUF_context> gguf_init_from_file(const char* fname)
{
    std::shared_ptr<GGUF_context> ctx = std::make_shared<GGUF_context>();

    // load function
    {
        FILE * file = gguf_fopen(fname, "rb");

        if (!file)
        {
            M_Error_(Error::Code::StsBadArg, ("Error to load file %s!\n", fname));
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
                    M_Error_(Error::Code::StsBadArg, ("%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]));
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
            ctx->tensors = nullptr;
            ctx->data  = nullptr;

            ok = ok & gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
            ok = ok & gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
            ok = ok & gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

            // Add code is compatible with version 1.
            if (ctx->header.version == 1)
            {
                M_Error_(Error::Code::StsBadArg, ("%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__));
                fclose(file);
                return nullptr;
            }

            // sanity-checks to prevent from integer/buffer overflows.
            ok = ok & (ctx->header.n_tensors < (SIZE_MAX/2)/sizeof(GGUF_tensor));
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
                                    M_Error_(Error::Code::StsError, ("%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__));
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
                                    M_Error_(Error::Code::StsNoMem, ("%s: array size is too large %d ! \n", __func__, (int)kv->value.arr.n));
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
                            default: M_Assert(false && "invalid type"); break;
                        }
                    } break;
                    default: M_Assert(false && "invalid type");
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
            ctx->tensors = (GGUF_tensor *)MMemoryAllocAlign(ctx->header.n_tensors * sizeof (GGUF_tensor));

            for (uint64_t i = 0; i < ctx->header.n_tensors; i++)
            {
                GGUF_tensor* tensors = &ctx->tensors[i];

                // set default dim
                for (int j = 0; j < GGML_MAX_DIMS; j++)
                {
                    tensors->ne[j] = 1;
                }

                ok = ok && gguf_fread_str(file, &tensors->name,  &offset);
                ok = ok && gguf_fread_el(file, &tensors->n_dims, sizeof(tensors->n_dims), &offset);

                ok = ok && (tensors->n_dims <= GGML_MAX_DIMS);

                // get right dim
                for (uint32_t j = 0; j < tensors->n_dims; ++j)
                {
                    ok = ok && gguf_fread_el(file, &tensors->ne[j], sizeof(tensors->ne[j]), &offset);
                }

                ok = ok && gguf_fread_el(file, &tensors->type, sizeof(tensors->type), &offset);
                ok = ok && gguf_fread_el(file, &tensors->offset, sizeof(tensors->offset), &offset);

                gguf_tensor_info_sanitize(tensors);

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
                GGUF_tensor* info = &ctx->tensors[i];

                const int64_t ne =
                        (int64_t) info->ne[0] * (int64_t) info->ne[1] *
                        (int64_t) info->ne[2] * (int64_t) info->ne[3];

                // need every data type
                int aa = ggml_blck_size(info->type);
                if (ne % aa != 0)
                {
                    fprintf(stderr, "%s: tensor '%s' of type %d (%s) number of elements (%" PRId64 ") is not a multiple of block size (%d)\n",
                            __func__, info->name.data, (int)info->type, ggml_type_name(info->type), ne, ggml_blck_size(info->type));
                    fclose(file);
                    ctx.reset();
                    return NULL;
                }

                const size_t size_cur = ggml_row_size(info->type, ne);
                ctx->size += M_PAD(size_cur, ctx->alignment);
            }
        }

        // loading tensor data
        {
            // Alloc memory
            size_t mem_size = M_PAD(ctx->size, GGML_MEM_ALIGN);
            ctx->data = MMemoryAllocAlign(mem_size);

            // loading all data from binary to data
            ok = ok & gguf_fread_el(file, ctx->data, ctx->size, &offset);

            // create and loading tensor one by one.
            for (int i = 0; i < ctx->header.n_tensors; i++)
            {
                struct GGUF_tensor* tensor = &ctx->tensors[i];

                size_t data_size = ggml_row_size(tensor->type, tensor->ne[0]);
                for (int i = 1; i < tensor->n_dims; i++)
                {
                    data_size *= tensor->ne[i];
                }

                tensor->nb[0] = ggml_type_size(tensor->type);
                tensor->nb[1] = tensor->nb[0] * (tensor->ne[0]/ ggml_blck_size(tensor->type));
                for (int i = 2; i < GGML_MAX_DIMS; i++)
                {
                    tensor->nb[i] = tensor->nb[i-1] * tensor->ne[i-1];
                }

                tensor->size = data_size;
                tensor->data = (char *)ctx->data + tensor->offset;
            }
        }

        fclose(file);
    }

    return ctx;
}

static
void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

static void llama_escape_whitespace(std::string & text)
{
    replace_all(text, " ", "\xe2\x96\x81");
}

    static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

    static size_t unicode_len_utf8(char src)
{
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct LLM_KV_Impl {
    LLM_KV_Impl(LLM_ARCH arch) : arch(arch) {}

    LLM_ARCH arch;

    std::string operator()(LLM_KV kv) const {
        return format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));
    }
};

struct LLama_param
{
    uint32_t n_vocab = 0;      // vocabulary length
    uint32_t n_ctx_length = 0; // context length
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    uint32_t n_layer = 0;
    uint32_t n_head_kv = 0; // n_head_kv is optional, by default it is equal to n_head
    uint32_t n_rope_dim_count = 0; //
    float rope_freq_base_train = 10000.f;

    // specific model
    float f_norm_rms_eps = 0.f;
};

struct LLama_loader
{
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t n_bytes = 0;

    LLama_param params;

    std::unique_ptr<LLama_file> file;
    std::unique_ptr<LLama_mmap> mapping;

    struct LLama_tensor_weights
    {
        uint16_t idx; // source file idex
        size_t offs;  // tensor data offset in the original file.
        GGUF_tensor* tensor;
        char name[GGML_MAX_NAME];

        // TODO here
        LLama_tensor_weights(uint16_t idx, const struct GGUF_context* gguf_ctx)
                : idx(idx)
        {
            tensor = &gguf_ctx->tensors[idx];
            offs = tensor->offset;

            for (int i = 0; i < GGML_MAX_NAME; i++)
            {
                name[i] = tensor->name.data[i];
            }
        }
    };

    // kv_overrides is used for user reset model hyper-params for fine-tuning the model effect.
    std::unordered_map<std::string, struct LLama_model_kv_override> kv_overrides;

    std::shared_ptr<GGUF_context> meta;

    std::string arch_name;
    LLM_KV_Impl llmKv = LLM_KV_Impl(LLM_ARCH_UNKNOWN);

    LLM_ARCH get_arch()
    {
        return llmKv.arch;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, const bool required = true) {
        const int kid = gguf_find_key(meta.get(), key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
                GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const enum LLM_KV kid, T & result, const bool required = true) {
        return get_arr_n(llmKv(kid), result, required);
    }

    template<typename T>
    bool get_key(const std::string & key, T & result, const bool required = true) {
        auto it = kv_overrides.find(key);

        const struct LLama_model_kv_override * override =
                it != kv_overrides.end() ? &it->second : nullptr;
        const bool found = GGUFMeta::GKV<T>::set(meta.get(), key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool get_key(const enum LLM_KV kid, T & result, const bool required = true) {
        return get_key(llmKv(kid), result, required);
    }

    GGUF_tensor* get_tensor(const std::string& name)
    {
        M_Assert(meta && "meta is empty!!");
        int n_tensors = gguf_get_n_tensors(meta.get());

        // create Mat based on tensor info
        for (int i = 0; i < n_tensors; i++)
        {
            auto& t = meta->tensors[i];
            if (std::strcmp(name.c_str(), t.name.data) == 0)
            {
                return &t;
            }
        }
        return nullptr;
    }

    Mat create_mat(const std::string& name, bool required = true)
    {
        Mat m;
        M_Assert(meta && "meta is empty!!");
        GGUF_tensor* t = get_tensor(name);
        if (!t)
        {
            if (required)
                M_Error_(Error::StsBadFunc, ("Can not found tensor with name = %s ! \n", name.c_str()));
            else
            {
                M_PRINT_DBG_(NULL, ("Can not found tensor with name = %s ! \n", name.c_str()));
            }

            return m;
        }

        // convert uint64 to int 32
        std::vector<int> dims(t->n_dims);
        for (int i = 0; i < dims.size(); i++)
        {
            dims[i] = t->ne[i];
        }

        M_Assert(t->data && "tensor data is empty!!");
        switch (t->type) {
            case GGML_TYPE_F32:
                m = Mat(dims, DT_32F, const_cast<void *>(t->data));
                break;
            case GGML_TYPE_F16:
                m = Mat(dims, DT_16F, const_cast<void *>(t->data));
                break;
            case GGML_TYPE_Q8_0:
                m = Mat(dims, DT_8U, const_cast<void *>(t->data));
                break;
            default:
                M_Error_(Error::Code::StsNullPtr, ("Fail to create mat with type = %d !!", (int )t->type));
        }
        return m;
    }

    // TODO support the param overrider p argument!
    // 这个类别只需要做到，能够自由的加载所有参数到内存，以及方便的获取key-value键值对就行。
    LLama_loader(const std::string& fname, bool use_mmap, const struct LLama_model_kv_override* param_overrides_p)
    {
        int trace = 0;
        M_Assert(param_overrides_p == nullptr && "Currently, do not support the param override!");
        if (getenv("MINFER_LOG_LEVEL"))
        {
            trace = atoi(getenv("MINFER_LOG_LEVEL"));
        }

        // 可以根据我后面的结构来调整此部分的键值设置？
        if (param_overrides_p != nullptr) {
            // 根据配置修改一些参数，作为后面的features
            for (const struct LLama_model_kv_override *p = param_overrides_p; p->key[0] != 0; p++) {
                kv_overrides.insert({std::string(p->key), *p});
            }
        }

        meta = gguf_init_from_file(fname.c_str());

        // get architecture
        get_key(llmKv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llmKv = LLM_KV_Impl(llm_arch_from_string(arch_name));

        // get params
        this->get_key(LLM_KV_VOCAB_SIZE, params.n_vocab, false) || this->get_arr_n(LLM_KV_TOKENIZER_LIST, params.n_vocab);
        this->get_key(LLM_KV_CONTEXT_LENGTH, params.n_ctx_length);
        this->get_key(LLM_KV_EMBEDDING_LENGTH, params.n_embd);
        this->get_key(LLM_KV_FEED_FORWARD_LENGTH, params.n_ff);
        this->get_key(LLM_KV_ATTENTION_HEAD_COUNT, params.n_head);

        params.n_head_kv = params.n_head;
        this->get_key(LLM_KV_ATTENTION_HEAD_COUNT_KV, params.n_head_kv);

        this->get_key(LLM_KV_BLOCK_COUNT, params.n_layer);

        params.n_rope_dim_count = (params.n_head == 0) ? 0 : params.n_embd / params.n_head;
        this->get_key(LLM_KV_ROPE_DIMENSION_COUNT, params.n_rope_dim_count, false);
        M_Assert(params.n_rope_dim_count == params.n_embd / params.n_head && "Invalid n_rope_dim_count!");

        this->get_key(LLM_KV_ROPE_FREQ_BASE, params.rope_freq_base_train, false);
    }
};

template<>
bool LLama_loader::get_key(const enum LLM_KV kid, enum llama_pooling_type & result, const bool required) {
    uint32_t tmp;
    const bool found = get_key(kid, tmp, required);
    if (found) {
        result = (enum llama_pooling_type) tmp;
    } else {
        result = LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    return found;
}

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_LLAMA);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN {
    LLM_TN(LLM_ARCH arch) : arch(arch) {}

    LLM_ARCH arch;

    std::string operator()(LLM_TENSOR tensor) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return LLM_TENSOR_NAMES.at(arch).at(tensor);
    }

    std::string operator()(LLM_TENSOR tensor, const std::string suffix) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return LLM_TENSOR_NAMES.at(arch).at(tensor) + "." + suffix;
    }

    std::string operator()(LLM_TENSOR tensor, int bid) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid);
    }

    std::string operator()(LLM_TENSOR tensor, const std::string suffix, int bid) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid) + "." + suffix;
    }

    std::string operator()(LLM_TENSOR tensor, const std::string & suffix, int bid, int xid) const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }
        return format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid, xid) + "." + suffix;
    }
};

void readGGUF(const std::string path, std::vector<std::shared_ptr<LayerParams> >& netParams, std::shared_ptr<GGUF_Vocab>& gguf_vocab)
{
    netParams.clear();

    // different platform has different structure?
    // how to construct the model from context

    // TODO: 目前会通过loader去持有内存，从而让内存在create net的阶段可读。
    static LLama_loader loader = LLama_loader(path, false, nullptr);
    LLM_ARCH arch = loader.get_arch();

    std::cout<<"Arch = "<<LLM_ARCH_NAMES.at(arch)<<std::endl;
    // print base info of llm model
    std::cout<<"n_vocab = "<<loader.params.n_vocab<<std::endl;
    std::cout<<"n_ctx_length = "<<loader.params.n_ctx_length<<std::endl;
    std::cout<<"n_embd = "<<loader.params.n_embd<<std::endl;
    std::cout<<"n_ff = "<<loader.params.n_ff<<std::endl;
    std::cout<<"n_head = "<<loader.params.n_head<<std::endl;
    std::cout<<"n_head_kv = "<<loader.params.n_head_kv<<std::endl;
    std::cout<<"n_layer = "<<loader.params.n_layer<<std::endl;
    std::cout<<"n_rope_dim_count = "<<loader.params.n_rope_dim_count<<std::endl;
    std::cout<<"rope_freq_base_train = "<<loader.params.rope_freq_base_train<<std::endl;
    std::cout<<"f_norm_rms_eps = "<<loader.params.f_norm_rms_eps<<std::endl;

    gguf_vocab->loadFromGGUF(&loader);

    const auto getTensorName = LLM_TN(arch);
    // construct llama by loader to netParams
    std::string miss = "__missing__";

    auto& p = loader.params;
    // parse llama model
    if (arch == LLM_ARCH_LLAMA)
    {
        std::string out;

        // handle tok_embedding
        Mat embdMat = loader.create_mat(getTensorName(LLM_TENSOR_TOKEN_EMBD, "weight"));
        M_Assert(!embdMat.empty() && "Error when to create llama mat!");

        // set model input
        netParams.push_back(
                std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {0}, {1}))
                );

        netParams.push_back(
                std::shared_ptr<LayerParams>(
                new EmbeddingLayerParams({1}, {2}, p.n_vocab, p.n_embd, embdMat)));

        int layer_id = 2;
        // handle multi attention layer
        {
            for (int i = 0; i < loader.params.n_layer; i++)
            {
                // get attn Mats
                Mat attn_norm = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_NORM, "weight", i));
                Mat wq = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_Q, "weight", i));
                Mat wk = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_K, "weight", i));
                Mat wv = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_V, "weight", i));
                Mat wo = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_OUT, "weight", i));

                // optional bias tensors
                Mat bq = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_Q, "bias", i), false);
                Mat bk = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_K, "bias", i), false);
                Mat bv = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_V, "bias", i), false);
                Mat bo = loader.create_mat(getTensorName(LLM_TENSOR_ATTN_OUT, "bias", i), false);

                // add Attn layer
                netParams.push_back(
                        std::shared_ptr<LayerParams>(new AttentionLayerParams(
                                {layer_id}, {layer_id + 1}, p.n_ctx_length, p.n_embd, p.n_head, p.n_head_kv,
                                p.f_norm_rms_eps, attn_norm, wq, wk, wv, wo, bq, bk, bv, bo
                                ))
                        );

                layer_id ++;

                // get FFN mats
                Mat ffn_norm = loader.create_mat(getTensorName(LLM_TENSOR_FFN_NORM, "weight", i));
                Mat ffn_gate = loader.create_mat(getTensorName(LLM_TENSOR_FFN_GATE, "weight", i), false);
                Mat ffn_down = loader.create_mat(getTensorName(LLM_TENSOR_FFN_DOWN, "weight", i), false);
                Mat ffn_up = loader.create_mat(getTensorName(LLM_TENSOR_FFN_UP, "weight", i), false);

                // add FFN layer
                netParams.push_back(
                        std::shared_ptr<LayerParams>(new FeedForwardLayerParams(
                                {layer_id}, {layer_id + 1}, ActivateType::SILU, p.n_embd, p.n_ff, p.f_norm_rms_eps,
                                ffn_norm, ffn_gate, ffn_up, ffn_down
                                ))
                        );
                layer_id++;
            }
        }

        // handle output
        {
            // create output norm
            Mat out_norm = loader.create_mat(getTensorName(LLM_TENSOR_OUTPUT_NORM, "weight"));
            M_Assert(!out_norm.empty() && "Error when to create llama mat!");

            netParams.push_back(std::shared_ptr<LayerParams>(
                    new RMSNormLayerParams({layer_id}, {layer_id + 1}, p.n_embd, p.f_norm_rms_eps, out_norm)));

            layer_id++;

            Mat outWeight = loader.create_mat(getTensorName(LLM_TENSOR_OUTPUT, "weight"), false);

            // if output is NULL, init from the input tok embed
            if (outWeight.empty())
            {
                outWeight = loader.create_mat(getTensorName(LLM_TENSOR_TOKEN_EMBD, "weight"));
            }

            // create output out-embedding
            netParams.push_back(std::shared_ptr<LayerParams>(
                    new LinearLayerParams({layer_id}, {layer_id + 1}, p.n_embd, p.n_vocab, outWeight)));
            layer_id++;
        }

        // set model output
        netParams.push_back(
                std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {layer_id}, {layer_id + 1}))
        );
    }
}

/* *********************************************************************************************************************
 *                                                >>>   Tokenizer   <<<
 * *********************************************************************************************************************
 */

GGUF_Vocab::GGUF_Vocab()
{
}

GGUF_Vocab::~GGUF_Vocab()
{

}

typedef int llama_vocab;

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

bool GGUF_Vocab::loadFromGGUF(void* _loader)
{
    LLama_loader* loader = (LLama_loader*)_loader;
    M_Assert(loader && "loader is null!");

    std::shared_ptr<GGUF_context> ctx = loader->meta;
    if (!ctx)
    {
        M_Error(NULL, "GGUF context is null");
        return false;
    }

    // determine the vocabulary type
    {
        std::string tokenizer_model;
        std::string tokenizer_pre;

        loader->get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);
        // loader->get_key(LLM_KV_TOKENIZER_PRE,   tokenizer_pre, false);
        loader->get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, n_token_types, false);

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none")
        {
            type = LLAMA_VOCAB_TYPE_NONE;

            // default special tokens
            special_bos_id  = LLAMA_TOKEN_NULL;
            special_eos_id  = LLAMA_TOKEN_NULL;
            special_unk_id  = LLAMA_TOKEN_NULL;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
            linefeed_id     = LLAMA_TOKEN_NULL;

            // read vocab size from metadata
            uint32_t n_tokens = 0;
            if (loader->get_key(LLM_KV_VOCAB_SIZE, n_tokens, false))
            {
                M_PRINT_DBG_(NULL, ("%s: adding %u dummy tokens\n", __func__, n_tokens));

                id_to_token.resize(n_tokens);
            }

            return true;
        }

        if (tokenizer_model == "llama") {
            type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            special_bos_id  = 1;
            special_eos_id  = 2;
            special_unk_id  = 0;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
        }
        else
        {
            M_Error_(NULL, ("Unsupported tokenizer model: %s", tokenizer_model.c_str()));
        }

        if (type == LLAMA_VOCAB_TYPE_BPE)
        {
            M_Warning_(NULL, ("%s: missing pre-tokenizer type, using: 'default'\n", __func__));
            M_Warning_(NULL, ("%s:                                             \n", __func__));
            M_Warning_(NULL, ("%s: ************************************        \n", __func__));
            M_Warning_(NULL, ("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__));
            M_Warning_(NULL, ("%s: CONSIDER REGENERATING THE MODEL             \n", __func__));
            M_Warning_(NULL, ("%s: ************************************        \n", __func__));
            M_Warning_(NULL, ("%s:                                             \n", __func__));
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        } else if (type == LLAMA_VOCAB_TYPE_SPM) {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = true;
            clean_spaces = false;
            add_bos = true;
            add_eos = false;
        } else if (type == LLAMA_VOCAB_TYPE_WPM) {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = false;
            clean_spaces = true;
            add_bos = true;
            add_eos = false;
        // } else if (type == LLAMA_VOCAB_TYPE_UGM) {
        //     pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        //     add_bos = false;
        //     add_eos = true;
        // } else if (type == LLAMA_VOCAB_TYPE_RWKV) {
        //     pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        //     add_space_prefix = false;
        //     clean_spaces = false;
        //     add_bos = false;
        //     add_eos = false;
        } else {
            M_Error_(NULL, ("Unsupported tokenizer type: %d", (int)type));
        }

        loader->get_key(LLM_KV_TOKENIZER_ADD_PREFIX,      add_space_prefix,         false);
        // ml.get_key(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS, remove_extra_whitespaces, false);
    }

    
    const int token_idx = gguf_find_key(ctx.get(), LLM_KV_NAMES.at(LLM_KV_TOKENIZER_LIST));
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float * scores = nullptr;
    const int score_idx = gguf_find_key(ctx.get(), LLM_KV_NAMES.at(LLM_KV_TOKENIZER_SCORES));
    if (score_idx != -1) {
        scores = (const float * ) gguf_get_arr_data(ctx.get(), score_idx);
    }

    
    const int * toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx.get(), LLM_KV_NAMES.at(LLM_KV_TOKENIZER_TOKEN_TYPE));
    if (toktype_idx != -1) {
        toktypes = (const int * ) gguf_get_arr_data(ctx.get(), toktype_idx);
    }

    uint32_t n_tokens = gguf_get_arr_n(ctx.get(), token_idx);
    id_to_token.resize(n_tokens);

    // load tokens into memory
    for (uint32_t i = 0; i < n_tokens; i++) 
    {
        std::string word = gguf_get_arr_str(ctx.get(), token_idx, i);
        if (word.empty()) 
        {
            M_Warning_(NULL, ("%s: empty token at index %u\n", __func__, i));
            word = "[EMPTY_" + std::to_string(i) + "]";
        }

        token_to_id[word] = i;
        max_token_len = std::max(max_token_len, (int) word.size());

        auto & token_data = id_to_token[i];
        token_data.text  = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.attr  = LLAMA_TOKEN_ATTR_NORMAL;

        if (toktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
            switch(toktypes[i]) 
            {
                case LLAMA_TOKEN_TYPE_UNKNOWN:      token_data.attr = LLAMA_TOKEN_ATTR_UNKNOWN;      break;
                case LLAMA_TOKEN_TYPE_UNUSED:       token_data.attr = LLAMA_TOKEN_ATTR_UNUSED;       break;
                case LLAMA_TOKEN_TYPE_NORMAL:       token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;       break;
                case LLAMA_TOKEN_TYPE_CONTROL:      token_data.attr = LLAMA_TOKEN_ATTR_CONTROL;      break;
                case LLAMA_TOKEN_TYPE_USER_DEFINED: token_data.attr = LLAMA_TOKEN_ATTR_USER_DEFINED; break;
                case LLAMA_TOKEN_TYPE_BYTE:         token_data.attr = LLAMA_TOKEN_ATTR_BYTE;         break;
                case LLAMA_TOKEN_TYPE_UNDEFINED:    token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
                default:                            token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }

    M_Assert(id_to_token.size() == token_to_id.size() && "vocab size not match!");


    // build special tokens cache
    {
        for (llama_token id = 0; id < (llama_token) n_tokens; ++id)
        {
            if (id_to_token[id].attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN))
            {
                cache_special_tokens.push_back(id);
            }
        }

        std::sort(cache_special_tokens.begin(), cache_special_tokens.end(),
            [&] (const llama_token a, const llama_token b) {
                return id_to_token[a].text.size() > id_to_token[b].text.size();
            }
        );

        M_PRINT_DBG_(NULL, ("%s: special tokens cache size = %u\n", __func__, (uint32_t) cache_special_tokens.size()));
    }
    // load
    // Load vocabulary from GGUF context
    return true;
}

void GGUF_Vocab::decode(const std::vector<int> &out_ids, std::string &out_text)
{
    // Implement the decoding logic here
    for (int id : out_ids)
    {
        if (id >= 0 && id < (int)id_to_token.size())
        {
            out_text += id_to_token[id].text;
        }
    }

    llama_unescape_whitespace(out_text);
    // 替换所有前导下划线为空格
    for (size_t i = 0; i < out_text.size(); ++i) {
        if (out_text[i] == '_')
            out_text[i] = ' ';
    }
    // 去掉开头可能多余的空格
    if (!out_text.empty() && out_text[0] == ' ') out_text.erase(0, 1);
}

struct llm_bigram_spm {
struct comparator {
    bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
        return (l.score < r.score) || (l.score == r.score && l.left > r.left);
    }
};
using queue_storage = std::vector<llm_bigram_spm>;
using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
llm_symbol::index left;
llm_symbol::index right;
float score;
size_t size;
};

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_token _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_token) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
        M_Assert(_offset >= 0);
        M_Assert(_length >= 1);
        M_Assert(offset + length <= raw_text.length());
    }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_token token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

class LLM_tokenizer_spm
{
public:
    LLM_tokenizer_spm(std::unordered_map<std::string, llama_token>& _token_to_id, std::vector<token_data>& _id_to_token,  std::vector<llama_token>& _special_tokens,
    bool _add_space_prefix,
    bool _add_bos,
    bool _add_eos,
    bool _ignore_merges,
    bool _clean_spaces,
    bool _remove_extra_whitespaces,
    bool _escape_whitespaces,
    bool _treat_whitespace_as_suffix,
    llama_token _special_bos_id,
    llama_token _special_eos_id
    )
    : token_to_id(_token_to_id), id_to_token(_id_to_token), cache_special_tokens(_special_tokens)
    , add_space_prefix(_add_space_prefix), add_bos(_add_bos), add_eos(_add_eos)
    , ignore_merges(_ignore_merges), clean_spaces(_clean_spaces)
    , remove_extra_whitespaces(_remove_extra_whitespaces)
    , escape_whitespaces(_escape_whitespaces), treat_whitespace_as_suffix(_treat_whitespace_as_suffix)
    , special_bos_id(_special_bos_id), special_eos_id(_special_eos_id)
    {
        n_tokens = id_to_token.size();

        // token size should be the same
        M_Assert(id_to_token.size() == token_to_id.size() && "vocab size not match!");
    }

    ~LLM_tokenizer_spm()
    {
    }

    /* @biref  Tokenizer state partition
    * 输入文本要先被切分成一系列片段，这些片段可能是：
    * 普通字符串（例如 "hello"）
    * 特殊 token（例如 "<s>"，"</s>"）
    *
    */
    void tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special)
    {
        // for each special token
        for (const llama_token special_id : cache_special_tokens)
        {
            const auto & data = id_to_token[special_id];
            const auto & text = data.text;

            if (!parse_special && (data.attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN))) {
                // Ignore control and unknown tokens when parse_special == false
                continue;
                // User-defined tokens are still pre-tokenized before everything else
                // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
                // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
            }


        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                const auto & raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true)
                {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    auto match = raw_text.find(text, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

                    // check if match is within bounds of offset <-> length
                    if (match + text.length() > raw_text_base_offset + raw_text_base_length) break;

                    M_PRINT_DBG_(NULL, ("FF: (%ld %ld %ld) '%s'\n", raw_text.length(), raw_text_base_offset, raw_text_base_length, raw_text.substr(raw_text_base_offset, raw_text_base_length).c_str()));

                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & LLAMA_TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

                        M_PRINT_DBG_(NULL, ("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text.substr(left_reminder_offset, left_reminder_length).c_str()));
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + text.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + text.length());

                        if (data.attr & LLAMA_TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

                        M_PRINT_DBG_(NULL, ("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text.substr(right_reminder_offset, right_reminder_length).c_str()));

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

                        M_PRINT_DBG_(NULL, ("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text.substr(raw_text_base_offset, raw_text_base_length).c_str()));
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
        }
    }

    // 目前对于特殊字符不太考虑
    void tokenize(const std::string & text, std::vector<llama_token> & output)
    {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size())
        {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (int i = 1; i < (int) symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
                }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

    llama_token text_to_token(const std::string &text) const
    {
        auto it = token_to_id.find(text);
        if (it == token_to_id.end())
        {
            return LLAMA_TOKEN_NULL;
        }
        return it->second;
    }

    llama_token byte_to_token(uint8_t ch) const
    {
        static const char * hex = "0123456789ABCDEF";

        // case LLAMA_VOCAB_TYPE_SPM:
        // case LLAMA_VOCAB_TYPE_UGM:
        {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = token_to_id.find(buf);
            if (token != token_to_id.end())
            {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return token_to_id.at(buf2);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llama_token> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = text_to_token(text);

        // Do we need to support is_unused?
        if (token != LLAMA_TOKEN_NULL) {
            output.push_back(token);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_token id = byte_to_token(symbol.text[j]);
                output.push_back(id);
            }
            return;
        }

        resegment(symbols[p->second.first], output);
        resegment(symbols[p->second.second], output);
    }
    void try_add_bigram(int left, int right)
    {
        if (left == -1 || right == -1) {
            return;
        }
        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = text_to_token(text);

        if (token == LLAMA_TOKEN_NULL) {
            return;
        }

        if (static_cast<uint32_t>(token) >= n_tokens) {
            return;
        }

        M_Assert(token <= (int)id_to_token.size() && "token id out of range!");
        const auto & tok_data = id_to_token[token];

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    std::unordered_map<std::string, llama_token> token_to_id;
    std::vector<token_data>                      id_to_token;
    std::vector<llama_token> cache_special_tokens;
    bool add_space_prefix           = false;
    bool add_bos                    = false;
    bool add_eos                    = false;
    bool ignore_merges              = false;
    bool clean_spaces               = false;  // clean_up_tokenization_spaces
    bool remove_extra_whitespaces   = false;
    bool escape_whitespaces         = true;
    bool treat_whitespace_as_suffix = false;
    llama_token special_bos_id  = 1;
    llama_token special_eos_id  = 2;
    size_t n_tokens = 0;
    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;
    std::map<std::string, std::pair<int, int>> rev_merge;

};

// 目前只支持spm的编码
void GGUF_Vocab::encode(const std::string raw_text, std::vector<int> &output)
{
    output.clear();
    // 需要单独实现SPM encode

    /* 标志说明：
    *  如果 add_special = true → 把特殊 token 直接转成它们的文本，比如 "<s>"。
    *  如果 add_special = false → 遇到特殊 token 会被 忽略/跳过，只输出普通文本。
     */
    bool add_special = true;

    /*  如果 parse_special = true → 看到 "<s>", "</s>", <unk> 这样的字符串，会直接转成对应的 token ID。
     *  如果 parse_special = false → 会把 "<s>" 当成普通文本继续用 BPE/SentencePiece 分词。
     */
    bool parse_special = false;

    LLM_tokenizer_spm session(token_to_id, id_to_token, cache_special_tokens,
    add_space_prefix,
    add_bos,
    add_eos,
    ignore_merges,
    clean_spaces,
    remove_extra_whitespaces,
    escape_whitespaces,
    treat_whitespace_as_suffix,
    special_bos_id,
    special_eos_id
    );

    // std::vector<llama_token> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty())
    {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        session.tokenizer_st_partition(fragment_buffer, parse_special);
    }

      // case LLAMA_VOCAB_TYPE_SPM:
    {
        // OG tokenizer behavior:
        //
        // tokenizer.encode('', add_special_tokens=True)  returns [1]
        // tokenizer.encode('', add_special_tokens=False) returns []

        bool is_prev_special = true;  // prefix with space if first token

        if (add_special && add_bos) {
            M_Assert(special_bos_id != LLAMA_TOKEN_NULL);
            output.push_back(special_bos_id);
            is_prev_special = true;
        }

        for (const auto & fragment : fragment_buffer) {
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                std::string text;

                // prefix with space if previous is special
                if (add_space_prefix && is_prev_special) {
                    text = ' ';
                }

                text += fragment.raw_text.substr(fragment.offset, fragment.length);

                M_PRINT_DBG_(NULL, ("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str()));

                llama_escape_whitespace(text);

                session.tokenize(text, output);
                is_prev_special = false;
            } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                output.push_back(fragment.token);
                is_prev_special = true;
            }
        }

        if (add_special && add_bos && output.size() >= 2 && output[1] == special_bos_id) {
            M_Warning_(NULL,
                ("%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__));
        }

        if (add_special && add_eos) {
            M_Assert(special_eos_id != LLAMA_TOKEN_NULL);
            output.push_back(special_eos_id);
        }
    }
}

}