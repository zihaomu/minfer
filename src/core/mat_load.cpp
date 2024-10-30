//
// Created by mzh on 2024/1/31.
//

#include "minfer/mat.h"
#include "minfer/basic_op.h"
#include "minfer/system.h"
#include "minfer/utils.h"

namespace minfer
{

// The following code has taken from https://github.com/opencv/opencv/modules/dnn/test/npy_blob.cpp
static std::string getType(const std::string& header)
{
    std::string field = "'descr':";
    int idx = header.find(field);
    M_Assert(idx != -1);

    int from = header.find('\'', idx + field.size()) + 1;
    int to = header.find('\'', from);
    return header.substr(from, to - from);
}

static std::string getFortranOrder(const std::string& header)
{
    std::string field = "'fortran_order':";
    int idx = header.find(field);
    M_Assert(idx != -1);

    int from = header.find_last_of(' ', idx + field.size()) + 1;
    int to = header.find(',', from);
    return header.substr(from, to - from);
}

static std::vector<int> getShape(const std::string& header)
{
    std::string field = "'shape':";
    int idx = header.find(field);
    M_Assert(idx != -1);

    int from = header.find('(', idx + field.size()) + 1;
    int to = header.find(')', from);

    std::string shapeStr = header.substr(from, to - from);
    if (shapeStr.empty())
        return std::vector<int>(1, 1);

    // Remove all commas.
    shapeStr.erase(std::remove(shapeStr.begin(), shapeStr.end(), ','),
                   shapeStr.end());

    std::istringstream ss(shapeStr);
    int value;

    std::vector<int> shape;
    while (ss >> value)
    {
        shape.push_back(value);
    }
    return shape;
}

// TODO current only support the fp32 data
Mat readMatFromNpy(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::binary);
    M_Assert(ifs.is_open());

    std::string magic(6, '*');
    ifs.read(&magic[0], magic.size());
    M_Assert(magic == "\x93NUMPY");

    ifs.ignore(1);  // Skip major version byte.
    ifs.ignore(1);  // Skip minor version byte.

    unsigned short headerSize;
    ifs.read((char *) &headerSize, sizeof(headerSize));

    std::string header(headerSize, '*');
    ifs.read(&header[0], header.size());

    // Extract data type.
    M_Assert(getType(header) == "<f4");
    M_Assert(getFortranOrder(header) == "False");
    std::vector<int> shape = getShape(header);

    size_t typeSize = DT_ELEM_SIZE(DT_32F);
    Mat blob(shape, DT_32F);
    ifs.read((char *) blob.data, blob.total() * typeSize);
    M_Assert((size_t) ifs.gcount() == blob.total() * typeSize);

    return blob;
}

}