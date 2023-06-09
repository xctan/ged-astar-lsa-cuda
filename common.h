#ifndef __COMMON_H__
#define __COMMON_H__

#define panic(fmt, ...)                                                        \
    do {                                                                       \
        fprintf(stderr, "Panicked at %s:%d:\n", __FILE__, __LINE__);           \
        fprintf(stderr, fmt, ##__VA_ARGS__);                                   \
        exit(1);                                                               \
    } while (0)

#define my_assert(cond)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            panic("Assertion failed: %s\n", #cond);                            \
        }                                                                      \
    } while (0)

#define HANDLE_RESULT(result)                                                  \
    do {                                                                       \
        cudaError_t err = result;                                              \
        if (err != cudaSuccess) {                                              \
            panic("CUDA error code=%d(%s) \"%s\"\n", err,                      \
                  cudaGetErrorString(err), #result);                           \
        }                                                                      \
    } while (0)

typedef unsigned char u8;
typedef unsigned short u16;
typedef int i32;
typedef unsigned long long u64;

struct sibling {
    short weight;
    short image;
};

struct state {
    // previous state
    state *prev;
    // siblings
    sibling *siblings;
    // lowerbound
    int f;

    int mapped_cost;
    // image of this vertex in target graph
    u16 image;

    u16 level;
    // (unused) reference count
    short cs_cnt;

    short siblings_n;
};

#endif // __COMMON_H__