# reference:https://github.com/huggingface/huggingface.js/blob/main/packages/gguf/src/gguf.ts

import sys
from typing import Any
from enum import IntEnum

import numpy as np
import numpy.typing as npt

class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

def check_version(version):
    if version == 1 or version == 2 or version == 3:
        return True
    else:
        return False

# only little endian is supported! TODO add big endian
def data_get(
    data, offset: int, dtype: npt.DTypeLike, count: int = 1) -> npt.NDArray[Any]:
    count = int(count)
    itemsize = int(np.empty([], dtype = dtype).itemsize)
    end_offs = offset + itemsize * count
    return (
        data[offset:end_offs]
        .view(dtype = dtype)[:count]
        # .newbyteorder(override_order)
    )

def data_read_version_size(data, offset: int, version: int):
    if version == 1:
        return data_get(data, offset, np.uint32)[0], 4
    elif version == 2 or version == 3:
        return data_get(data, offset, np.uint64)[0], 8
    else:
        raise ValueError(f'Sorry, file appears to be version {version} which we cannot handle')

def data_read_string(data, offset: int, version: int):
    str_length, str_length_len = data_read_version_size(data, offset, version)
    byte = data[offset+int(str_length_len):offset+int(str_length_len)+int(str_length)]
    value = byte.tobytes().decode('utf-8')
    len = int(str_length_len + str_length)
    return value, len

def readMetadataValue(data, type, offset, version):
    if type == GGUFValueType.UINT8:
        return data_get(data, np.uint8)[0], 1
    elif type == GGUFValueType.INT8:
        return data_get(data, np.int8)[0], 1
    elif type == GGUFValueType.UINT16:
        return data_get(data, offset, np.uint16)[0], 2
    elif type == GGUFValueType.INT16:
        return data_get(data, offset, np.int16)[0], 2
    elif type == GGUFValueType.UINT32:
        return data_get(data, offset, np.uint32)[0], 4
    elif type == GGUFValueType.INT32:
        return data_get(data, offset, np.int32)[0], 4
    elif type == GGUFValueType.FLOAT32:
        return data_get(data, offset, np.float32)[0], 4
    elif type == GGUFValueType.BOOL:
        return data_get(data, offset, np.uint8)[0], 1
    elif type == GGUFValueType.STRING:
        return data_read_string(data, offset, version=version)
    elif type == GGUFValueType.ARRAY:
        typeArray = data_get(data, offset, np.uint32)
        typeLength = 4
        lengthArray, lengthLength = data_read_version_size(data, offset + typeLength, version=version)
        length = typeLength + lengthLength

        arrayValues = []
        for i in range(lengthArray):
            value, len = readMetadataValue(data, typeArray, offset= offset + length, version=version)
            arrayValues.append(value)
            length += len

        return arrayValues, length
    elif type == GGUFValueType.UINT64:
        return data_get(data, offset, np.uint64)[0], 8
    elif type == GGUFValueType.INT64:
        return data_get(data, offset, np.int64)[0], 8
    elif type == GGUFValueType.FLOAT64:
        return data_get(data, offset, np.float64)[0], 8
    else:
        raise ValueError(f'Sorry, un-supported GGUFValueType {type}!')

def parse_gguf(model_path):
    data = np.memmap(model_path, mode = 'r')

    offs = 0
    magic = data_get(data, offs, np.uint32).tobytes()
    print("magic", magic.decode('utf-8'))
    if (magic != b'GGUF'):
        print("is not gguf file")
        sys.exit(1)

    offs += 4
    version = data_get(data, offs, np.uint32)
    if not check_version(version):
        raise ValueError(f'Sorry, file appears to be version {version} which we cannot handle')

    print("version", version)
    offs += 4
    tensor_count, tensor_count_len = data_read_version_size(data, offs, version)
    offs += tensor_count_len
    kv_count, kv_count_len = data_read_version_size(data, offs, version)
    offs += kv_count_len

    print("tensor_count: ", tensor_count)
    print("kv_count: ", kv_count)

    metadata = {} # use dictionary to store parsed data.

    # parse gguf head info
    for i in range(kv_count):
        key, k_len = data_read_string(data, offs, version)
        offs += k_len
        
        type = data_get(data, offs, np.uint32)[0]
        offs += 4

        value, len = readMetadataValue(data, type, offs, version)
        if len > 100:
            print("i = ", i, ", k-v = ", key, ":", value[:100])
        else:
            print("i = ", i, ", k-v = ", key, ":", value)
        offs += len
        metadata[key] = value

    # parse gguf tensor info
    for i in range(tensor_count):
        key, k_len = data_read_string(data, offs, version)
        offs += k_len
        
        nDims = data_get(data, offs, np.uint32)[0]
        offs += 4

        dims = []
        for j in range(nDims):
            dim, dim_len = data_read_version_size(data, offs, version)
            offs += dim_len
            dims.append(dim)
        
        type = data_get(data, offs, np.uint32)[0]
        offs += 4

        tensorOffset = data_get(data, offs, np.uint64)[0]
        offs += 8

        print("tensor i = ", i, ", k = ", key, ", type = ", type, ", dims = ", dims, ", tensorOffset = ", tensorOffset)

if __name__ == '__main__':
    model_path = "/Users/mzh/work/models/llm_model/TinyLlama-1.1B-Chat-v1.0/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    parse_gguf(model_path)