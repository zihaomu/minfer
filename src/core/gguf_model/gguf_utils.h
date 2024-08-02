//
// Created by mzh on 2024/7/23.
//

#ifndef MINFER_GGUF_UTILS_H
#define MINFER_GGUF_UTILS_H

#include "minfer.h"
#include "../memory_utils.h"
#include "gguf_loader.h"

namespace minfer
{

#define GGML_MAX_NAME           64
#define GGML_MAX_DIMS           4
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGML_MEM_ALIGN 16
#define GGUF_MAGIC "GGUF"

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


//>>>>>>>>>>>>>>>>>>>>>> GGUF common  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

struct GGUF_header
{
    char magic[4];       // GGUF

    uint32_t version;
    uint64_t n_tensors; // kv tensor number, GGUV V2
    uint64_t n_kv;      // kv number
};

static size_t GGUF_TYPE_size(enum GGUF_TYPE type) {
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
struct GGUF_tensor {
    struct GGUF_str name;

    uint32_t n_dims;
    uint64_t  ne[GGML_MAX_DIMS];
    uint64_t  nb[GGML_MAX_DIMS];

    GGML_TYPE type;

    uint64_t offset;

    const void* data;
    size_t size;
};

struct GGUF_context {
    struct GGUF_header header;
    struct GGUF_kv *kv;           // pointer to all the kv list. What is kv? kv means the gguf key-value data struct, and the kv list contains all key-value info where the model has.
    struct GGUF_tensor* tensors;  // pointer to all tensor info list.

    size_t alignment;
    size_t offset;     // offset of data from beginning of file.
    size_t size;       // size of data in bytes

    void* data;

    ~GGUF_context();
};


//>>>>>>>>>>>>>>>>>>>>>> file  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
FILE* gguf_fopen(const char* fname, const char* mode);
bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset);

//>>>>>>>>>>>>>>>>>>>>>> get basic data  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
int ggml_blck_size(enum GGML_TYPE type);
size_t ggml_type_size(enum GGML_TYPE type);
const char * ggml_type_name(enum GGML_TYPE type);
size_t ggml_row_size(enum GGML_TYPE type, int64_t ne);

// gguf
size_t gguf_type_size(enum GGUF_TYPE type);
bool gguf_fread_str(FILE* file, GGUF_str* p, size_t* offset);

// gguf get data
const char * gguf_get_key(const struct GGUF_context * ctx, int key_id);
enum GGUF_TYPE gguf_get_kv_type(const struct GGUF_context * ctx, int key_id);
enum GGUF_TYPE gguf_get_arr_type(const struct GGUF_context * ctx, int key_id);
const void * gguf_get_arr_data(const struct GGUF_context * ctx, int key_id);
const char * gguf_get_arr_str(const struct GGUF_context * ctx, int key_id, int i);
int gguf_get_arr_n(const struct GGUF_context * ctx, int key_id);
uint8_t gguf_get_val_u8(const struct GGUF_context * ctx, int key_id);
int8_t gguf_get_val_i8(const struct GGUF_context * ctx, int key_id);
uint16_t gguf_get_val_u16(const struct GGUF_context * ctx, int key_id);
int16_t gguf_get_val_i16(const struct GGUF_context * ctx, int key_id);
uint32_t gguf_get_val_u32(const struct GGUF_context * ctx, int key_id);
int32_t gguf_get_val_i32(const struct GGUF_context * ctx, int key_id);
float gguf_get_val_f32(const struct GGUF_context * ctx, int key_id);
uint64_t gguf_get_val_u64(const struct GGUF_context * ctx, int key_id);
int64_t gguf_get_val_i64(const struct GGUF_context * ctx, int key_id);
double gguf_get_val_f64(const struct GGUF_context * ctx, int key_id);
bool gguf_get_val_bool(const struct GGUF_context * ctx, int key_id);
const char * gguf_get_val_str(const struct GGUF_context * ctx, int key_id);
const void * gguf_get_val_data(const struct GGUF_context * ctx, int key_id);

//>>>>>>>>>>>>>>>>>>>>>> tensor related  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
int gguf_find_tensor(const struct GGUF_context * ctx, const char * name);
int gguf_get_n_tensors(const struct GGUF_context * ctx);
char * gguf_get_tensor_name(const struct GGUF_context * ctx, int i);

//>>>>>>>>>>>>>>>>>>>>>> GGUF_context related  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
const char * GGUF_TYPE_name(enum GGUF_TYPE type);
int gguf_get_version(const struct GGUF_context * ctx);
size_t gguf_get_alignment(const struct GGUF_context * ctx);
size_t gguf_get_data_offset(const struct GGUF_context * ctx);
void * gguf_get_data(const struct GGUF_context * ctx);
int gguf_get_n_kv(const struct GGUF_context * ctx);
const char * gguf_get_key(const struct GGUF_context * ctx, int key_id);
int gguf_find_key(const struct GGUF_context * ctx, const char * key);

//>>>>>>>>>>>>>>>>>>>>>> LLama_context related  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
struct LLama_context {
    struct GGUF_header *header;
    struct GGUF_kv *kv;
    struct GGUF_tensor *tensors;

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

LLM_ARCH llm_arch_from_string(const std::string & name);

//>>>>>>>>>>>>>>>>>>>>>> GGUF Key value system  <<<<<<<<<<<<<<<<<<<<<<<<<<<<
namespace GGUFMeta {
    template <typename T, GGUF_TYPE gt_, T (*gfun)(const GGUF_context *, const int)>
    struct GKV_Base_Type {
        static constexpr GGUF_TYPE gt = gt_;

        static T getter(const GGUF_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr GGUF_TYPE gt = GGUF_TYPE_STRING;

        static std::string getter(const GGUF_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const GGUF_TYPE gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
    public:
        static constexpr GGUF_TYPE gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const GGUF_context *ctx, const int k) {
            return ArrayInfo {
                    gguf_get_arr_type(ctx, k),
                    size_t(gguf_get_arr_n(ctx, k)),
                    gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T>
    {
        GKV() = delete;

    public:
        static T get_kv(const GGUF_context * ctx, const int k)
        {
            const enum GGUF_TYPE kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                                                gguf_get_key(ctx, k), GGUF_TYPE_name(kt), GGUF_TYPE_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const LLama_model_kv_override_type ty)
        {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
            }
            return "unknown";
        }

        static bool validate_override(const LLama_model_kv_override_type expected_type, const struct LLama_model_kv_override * ovrd)
        {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                M_PRINT("%s: Using metadata override (%5s) '%s' = ",
                               __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        M_PRINT("%s\n", ovrd->bool_value ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        M_PRINT("%" PRId64 "\n", ovrd->int_value);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        M_PRINT("%.6f\n", ovrd->float_value);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                                format("Unsupported attempt to override %s type for metadata key %s\n",
                                       override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            M_PRINT("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                           __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct LLama_model_kv_override * ovrd)
        {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->bool_value;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct LLama_model_kv_override * ovrd)
        {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->int_value;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct LLama_model_kv_override * ovrd)
        {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->float_value;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct LLama_model_kv_override * ovrd)
        {
            (void)target;
            (void)ovrd;
            if (!ovrd) { return false; }
            // Currently, we should never end up here so it would be a bug if we do.
            throw std::runtime_error(format("Unsupported attempt to override string type for metadata key %s\n",
                                            ovrd ? ovrd->key : "NULL"));
        }

        static bool set(const GGUF_context * ctx, const int k, T & target, const struct LLama_model_kv_override * ovrd = nullptr)
        {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const GGUF_context * ctx, const char * key, T & target, const struct LLama_model_kv_override * ovrd = nullptr)
        {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const GGUF_context * ctx, const std::string & key, T & target, const struct LLama_model_kv_override * ovrd = nullptr)
        {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

} // namespace minfer

#endif //MINFER_GGUF_UTILS_H
