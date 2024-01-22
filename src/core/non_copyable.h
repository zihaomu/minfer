//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_NON_COPYABLE_H
#define MINFER_NON_COPYABLE_H

namespace minfer
{

/** protocol class. used to delete assignment operator. */
class NonCopyable {
public:
    NonCopyable()                    = default;
    NonCopyable(const NonCopyable&)  = delete;
    NonCopyable(const NonCopyable&&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&&) = delete;
};

}


#endif //MINFER_NON_COPYABLE_H
