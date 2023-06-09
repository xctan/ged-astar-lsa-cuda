#ifndef LIST_H
#define LIST_H

#include "common.h"

struct list {
    int length;
    int capacity;
    state **arr;
};

list **lists_create(int lists, int capacity);

list *list_create(int capacity);

void lists_destroy(list **lists_gpu, int lists);

void list_destroy(list *list);

__device__ void list_clear(list *list);

__device__ void list_insert(list *list, state *state);

__device__ void list_remove(list *list, int index);

__device__ state *list_get(list *list, int index);

#endif