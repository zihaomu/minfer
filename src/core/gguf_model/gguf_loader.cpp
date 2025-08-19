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
            M_ERROR(NULL, "warning: UnmapViewOfFile failed: llama release error in Windows!\n");
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
            ctx->tensors = nullptr;
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
                M_PRINT_DBG("Can not found tensor with name = %s ! \n", name.c_str());
            return m;
        }

        // convert uint64 to int 32
        std::vector<int> dims(t->n_dims);
        for (int i = 0; i < dims.size(); i++)
        {
            dims[i] = t->ne[i];
        }

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
                M_ERROR("Fail to create mat with type = %d !!", (int )t->type);
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

void readGGUF(const std::string path, std::vector<std::shared_ptr<LayerParams> >& netParams)
{
    netParams.clear();

    // different platform has different structure?
    // how to construct the model from context

    LLama_loader loader = LLama_loader(path, false, nullptr);
    LLM_ARCH arch = loader.get_arch();

    std::cout<<"Arch = "<<LLM_ARCH_NAMES.at(arch)<<std::endl;

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

            Mat outEmbd = loader.create_mat(getTensorName(LLM_TENSOR_OUTPUT, "weight"), false);

            // if output is NULL, init from the input tok embed
            if (outEmbd.empty())
            {
                outEmbd = loader.create_mat(getTensorName(LLM_TENSOR_TOKEN_EMBD, "weight"));
            }

            // create output out-embedding
            netParams.push_back(std::shared_ptr<LayerParams>(
                    new EmbeddingLayerParams({layer_id}, {layer_id + 1}, p.n_vocab, p.n_embd, outEmbd)));
            layer_id++;
        }

        // set model output
        netParams.push_back(
                std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {layer_id}, {layer_id + 1}))
        );
    }
}

}