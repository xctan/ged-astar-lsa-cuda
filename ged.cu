#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <map>
#include <cstdio>
#include <cassert>

#include "common.h"
#include "graph.h"
#include "heap.h"
#include "list.h"

#define STATES (32 * 1024ll * 1024)
#define MAX_N 64

__global__ void init_heap(heap **Q, state *states_pool);
__global__ void clear_list(list *S);
__global__ void fill_list(int k, heap **Q, list *S, state *states_pool);
__global__ void push_to_queues(int k, heap **Q, list *S, int off);
__device__ int f(const state *x);
__device__ int calculate_id();
__device__ state *state_create(int, int, int, state *prev, state *states_pool);
__device__ void sort_siblings(sibling *begin, sibling *end);
__device__ sibling *alloc_siblings(int n);

void states_pool_create(state **states);
void states_pool_destroy(state *states_pool);

#define THREADS_PER_BLOCK  128
#define BLOCKS 16

__device__ int total_Q_size = 0;
__device__ int out_of_memory = 0;
__device__ long long upper_bound = 65535;
// __device__ state *result = NULL;

__device__ int num_elabels, num_vlabels;
__device__ int q_n, *q_starts, *q_edges, *q_vlabels, *q_elabels;
__device__ int g_n, *g_starts, *g_edges, *g_vlabels, *g_elabels;
__device__ u8 *q_matrix;
__device__ int *mapping_order;
__device__ sibling *sibling_pool;
__device__ int used_siblings = 0;

struct pair { double weight; int index; };

void heap_top_down(int idx, int heap_n, std::vector<pair> &heap, int *pos) {
    pair tmp = heap[idx];
    while (2 * idx + 1 < heap_n) {
        int i = 2 * idx +1 ;
        if (i + 1 < heap_n && heap[i + 1].weight > heap[i].weight) ++i;
        if (heap[i].weight > tmp.weight) {
            heap[idx] = heap[i];
            pos[heap[idx].index] = idx;
            idx = i;
        }
        else break;
    }
    heap[idx] = tmp;
    pos[tmp.index] = idx;
}

void heap_bottom_up(int idx, std::vector<pair> &heap, int *pos) {
    pair tmp = heap[idx];
    while (idx > 0) {
        int i = (idx - 1) / 2;
        if (heap[i].weight < tmp.weight) {
            heap[idx] = heap[i];
            pos[heap[idx].index] = idx;
            idx = i;
        }
        else break;
    }
    heap[idx] = tmp;
    pos[tmp.index] = idx;
}


int ged_astar_gpu(const Graph &q, const Graph &g) {
    int k = THREADS_PER_BLOCK * BLOCKS;

    auto start = std::chrono::high_resolution_clock::now();

    // preprocessing
    {
        // TODO: swap q and g if q.num_vertices > g.num_vertices

        std::map<int, int> vlabel_map, elabel_map;
        for (int i = 0; i < q.num_vertices; i++) {
            if (vlabel_map.find(q.vertex_labels[i]) == vlabel_map.end()) {
                vlabel_map[q.vertex_labels[i]] = vlabel_map.size();
            }
        }
        for (int i = 0; i < q.num_edges; i++) {
            if (elabel_map.find(q.edge_labels[i]) == elabel_map.end()) {
                elabel_map[q.edge_labels[i]] = elabel_map.size();
            }
        }
        for (int i = 0; i < g.num_vertices; i++) {
            if (vlabel_map.find(g.vertex_labels[i]) == vlabel_map.end()) {
                vlabel_map[g.vertex_labels[i]] = vlabel_map.size();
            }
        }
        for (int i = 0; i < g.num_edges; i++) {
            if (elabel_map.find(g.edge_labels[i]) == elabel_map.end()) {
                elabel_map[g.edge_labels[i]] = elabel_map.size();
            }
        }
        int *q_vlabels_cpu = new int[q.num_vertices];
        int *q_elabels_cpu = new int[q.num_edges];
        int *g_vlabels_cpu = new int[g.num_vertices];
        int *g_elabels_cpu = new int[g.num_edges];
        for (int i = 0; i < q.num_vertices; i++) {
            q_vlabels_cpu[i] = vlabel_map[q.vertex_labels[i]];
        }
        for (int i = 0; i < q.num_edges; i++) {
            q_elabels_cpu[i] = elabel_map[q.edge_labels[i]];
        }
        for (int i = 0; i < g.num_vertices; i++) {
            g_vlabels_cpu[i] = vlabel_map[g.vertex_labels[i]];
        }
        for (int i = 0; i < g.num_edges; i++) {
            g_elabels_cpu[i] = elabel_map[g.edge_labels[i]];
        }
        int num_vlabels_cpu = vlabel_map.size();
        int num_elabels_cpu = elabel_map.size();

        // dump graphs
        printf("g_starts: ");
        for (int i = 0; i < g.num_vertices + 1; i++) {
            printf("%d ", g.edge_from_sep[i]);
        }
        printf("\ng_vlabels: ");
        for (int i = 0; i < g.num_vertices; i++) {
            printf("%d ", g_vlabels_cpu[i]);
        }
        printf("\nq_starts: ");
        for (int i = 0; i < q.num_vertices + 1; i++) {
            printf("%d ", q.edge_from_sep[i]);
        }
        printf("\nq_vlabels: ");
        for (int i = 0; i < q.num_vertices; i++) {
            printf("%d ", q_vlabels_cpu[i]);
        }
        printf("\n");

        HANDLE_RESULT(cudaMemcpyToSymbol(num_vlabels, &num_vlabels_cpu, sizeof(int)));
        HANDLE_RESULT(cudaMemcpyToSymbol(num_elabels, &num_elabels_cpu, sizeof(int)));
        HANDLE_RESULT(cudaMemcpyToSymbol(q_n, &q.num_vertices, sizeof(int)));

        void *q_starts_gpu, *q_edges_gpu, *q_vlabels_gpu, *q_elabels_gpu;
        HANDLE_RESULT(cudaMalloc(&q_starts_gpu, (q.num_vertices + 1) * sizeof(int)));
        HANDLE_RESULT(cudaMalloc(&q_edges_gpu, q.num_edges * sizeof(int)));
        HANDLE_RESULT(cudaMalloc(&q_vlabels_gpu, q.num_vertices * sizeof(int)));
        HANDLE_RESULT(cudaMalloc(&q_elabels_gpu, q.num_edges * sizeof(int)));

        HANDLE_RESULT(cudaMemcpyToSymbol(q_starts, &q_starts_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpyToSymbol(q_edges, &q_edges_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpyToSymbol(q_vlabels, &q_vlabels_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpyToSymbol(q_elabels, &q_elabels_gpu, sizeof(void *)));

        HANDLE_RESULT(cudaMemcpy(q_starts_gpu, q.edge_from_sep, (q.num_vertices + 1) * sizeof(int), cudaMemcpyDefault));
        HANDLE_RESULT(cudaMemcpy(q_edges_gpu, q.edge_to, q.num_edges * sizeof(int), cudaMemcpyDefault));
        HANDLE_RESULT(cudaMemcpy(q_vlabels_gpu, q_vlabels_cpu, q.num_vertices * sizeof(int), cudaMemcpyDefault));
        HANDLE_RESULT(cudaMemcpy(q_elabels_gpu, q_elabels_cpu, q.num_edges * sizeof(int), cudaMemcpyDefault));

        HANDLE_RESULT(cudaMemcpyToSymbol(g_n, &g.num_vertices, sizeof(int)));

        void *g_starts_gpu, *g_edges_gpu, *g_vlabels_gpu, *g_elabels_gpu;
        HANDLE_RESULT(cudaMalloc(&g_starts_gpu, (g.num_vertices + 1) * sizeof(int)));
        HANDLE_RESULT(cudaMalloc(&g_edges_gpu, g.num_edges * sizeof(int)));
        HANDLE_RESULT(cudaMalloc(&g_vlabels_gpu, g.num_vertices * sizeof(int)));
        HANDLE_RESULT(cudaMalloc(&g_elabels_gpu, g.num_edges * sizeof(int)));

        HANDLE_RESULT(cudaMemcpyToSymbol(g_starts, &g_starts_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpyToSymbol(g_edges, &g_edges_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpyToSymbol(g_vlabels, &g_vlabels_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpyToSymbol(g_elabels, &g_elabels_gpu, sizeof(void *)));

        HANDLE_RESULT(cudaMemcpy(g_starts_gpu, g.edge_from_sep, (g.num_vertices + 1) * sizeof(int), cudaMemcpyDefault));
        HANDLE_RESULT(cudaMemcpy(g_edges_gpu, g.edge_to, g.num_edges * sizeof(int), cudaMemcpyDefault));
        HANDLE_RESULT(cudaMemcpy(g_vlabels_gpu, g_vlabels_cpu, g.num_vertices * sizeof(int), cudaMemcpyDefault));
        HANDLE_RESULT(cudaMemcpy(g_elabels_gpu, g_elabels_cpu, g.num_edges * sizeof(int), cudaMemcpyDefault));

        u8 *q_matrix_cpu = new u8[q.num_vertices * q.num_vertices];
        for (int i = 0; i < q.num_vertices; i++) {
            u8 *row = q_matrix_cpu + i * q.num_vertices;
            for (int j = 0; j < q.num_vertices; j++) {
                row[j] = num_elabels_cpu;
            }
            for (int j = q.edge_from_sep[i]; j < q.edge_from_sep[i + 1]; j++) {
                row[q.edge_to[j]] = q_elabels_cpu[j];
            }
        }
        void *q_matrix_gpu;
        HANDLE_RESULT(cudaMalloc(&q_matrix_gpu, q.num_vertices * q.num_vertices * sizeof(u8)));
        HANDLE_RESULT(cudaMemcpyToSymbol(q_matrix, &q_matrix_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpy(q_matrix_gpu, q_matrix_cpu, q.num_vertices * q.num_vertices * sizeof(u8), cudaMemcpyDefault));
        delete[] q_matrix_cpu;

        int *mapping_order_cpu = new int[q.num_vertices];
        int *vlabels_cnt = new int[num_vlabels_cpu];
        int *elabels_cnt = new int[num_elabels_cpu];
        std::fill(vlabels_cnt, vlabels_cnt + num_vlabels_cpu, 0);
        std::fill(elabels_cnt, elabels_cnt + num_elabels_cpu, 0);
        for (int i = 0; i < g.num_vertices; i++) {
            vlabels_cnt[g_vlabels_cpu[i]]++;
            for (int j = g.edge_from_sep[i]; j < g.edge_from_sep[i + 1]; j++) {
                elabels_cnt[g_elabels_cpu[j]]++;
            }
        }

        printf("vlabels_cnt: ");
        for (int i = 0; i < num_vlabels_cpu; i++) {
            printf("%d ", vlabels_cnt[i]);
        }
        printf("\nelabels_cnt: ");
        for (int i = 0; i < num_elabels_cpu; i++) {
            printf("%d ", elabels_cnt[i]);
        }
        printf("\n");
        
        int root = 0;
        double root_weight = 0.0;
        for (int i = 0; i < q.num_vertices; i++) {
            double weight = 1 - vlabels_cnt[q_vlabels_cpu[i]] / (double)g.num_vertices;
            for (int j = q.edge_from_sep[i]; j < q.edge_from_sep[i + 1]; j++) {
                weight += 1 - elabels_cnt[q_elabels_cpu[j]] / (double)g.num_edges;
            }
            if (weight > root_weight) {
                root = i;
                root_weight = weight;
            }
        }
        
        std::vector<pair> heap(q.num_vertices);
        for (int i = 0; i < q.num_vertices; i++) {
            if (i == root) heap[i].weight = root_weight;
            else heap[i].weight = 0.0;
            heap[i].index = i;
        }
        std::swap(heap[0], heap[root]);
        int *pos = new int[q.num_vertices];
        for (int i = 0; i < q.num_vertices; i++) {
            pos[heap[i].index] = i;
        }
        int heap_n = q.num_vertices;
        for (int i = 0; i < q.num_vertices; i++) {
            int u = heap[0].index;
            mapping_order_cpu[i] = u;
            pos[u] = heap_n - 1;
            heap[0] = heap[--heap_n];
            pos[heap[0].index] = 0;
            //std::make_heap(heap.begin(), heap.begin() + heap_n, [](pair a, pair b) { return a.weight < b.weight; });
            heap_top_down(0, heap_n, heap, pos);
            for (int j = q.edge_from_sep[u]; j < q.edge_from_sep[u + 1]; j++) {
                if (pos[q.edge_to[j]] < heap_n) {
                    int idx = pos[q.edge_to[j]];
                    if (heap[idx].weight < 1e-5) {
                        heap[idx].weight += 1 - vlabels_cnt[q_vlabels_cpu[q.edge_to[j]]] / (double)g.num_vertices;
                    }
                    heap[idx].weight += 1 - elabels_cnt[q_elabels_cpu[j]] / (double)g.num_edges;
                    heap_bottom_up(idx, heap, pos);
                }
            }
            //std::make_heap(heap.begin(), heap.begin() + heap_n, [](pair a, pair b) { return a.weight < b.weight; });
        }
        printf("mapping order: ");
        for (int i = 0; i < q.num_vertices; i++) {
            printf("%d ", mapping_order_cpu[i]);
        }
        printf("\n");
        void *mapping_order_gpu;
        HANDLE_RESULT(cudaMalloc(&mapping_order_gpu, q.num_vertices * sizeof(int)));
        HANDLE_RESULT(cudaMemcpyToSymbol(mapping_order, &mapping_order_gpu, sizeof(void *)));
        HANDLE_RESULT(cudaMemcpy(mapping_order_gpu, mapping_order_cpu, q.num_vertices * sizeof(int), cudaMemcpyHostToDevice));
        delete[] mapping_order_cpu;

        delete[] q_vlabels_cpu;
        delete[] q_elabels_cpu;
        delete[] g_vlabels_cpu;
        delete[] g_elabels_cpu;

        void *sibling_pool_gpu;
        HANDLE_RESULT(cudaMalloc(&sibling_pool_gpu, STATES * sizeof(sibling)));
        HANDLE_RESULT(cudaMemcpyToSymbol(sibling_pool, &sibling_pool_gpu, sizeof(void *)));
    }

    heap **Q = heaps_create(k);
    list **Ss = lists_create(BLOCKS, 1000000);
    list *S = list_create(1024 * 1024);
    state *states_pool;
    states_pool_create(&states_pool);
    int total_Q_size_cpu;
    int out_of_memory_cpu;

    init_heap<<<1, 1>>>(Q, states_pool);
    int step = 0;
    do {
        clear_list<<<1, 1>>>(S);
        HANDLE_RESULT(cudaDeviceSynchronize());

        HANDLE_RESULT(cudaMemcpyFromSymbol(&total_Q_size_cpu, total_Q_size, sizeof(int)));
        printf("step %d, total_Q_size %d\n", step, total_Q_size_cpu);

        fill_list<<<BLOCKS, THREADS_PER_BLOCK>>>(k, Q, S, states_pool);
        HANDLE_RESULT(cudaMemcpyFromSymbol(&out_of_memory_cpu, out_of_memory, sizeof(int)));
        if (out_of_memory_cpu) break;
        HANDLE_RESULT(cudaDeviceSynchronize());
        push_to_queues<<<1, THREADS_PER_BLOCK>>>(k, Q, S, step);
        HANDLE_RESULT(cudaDeviceSynchronize());

        HANDLE_RESULT(cudaMemcpyFromSymbol(&total_Q_size_cpu, total_Q_size, sizeof(int)));
        printf("step %d, total_Q_size %d\n", step, total_Q_size_cpu);

        step++;
    } while (total_Q_size_cpu > 0);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = end - start;
    // output << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "\n";

    // TODO: output result

    {
        void *mapping_order_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&mapping_order_cpu, mapping_order, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&mapping_order_cpu, &mapping_order, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(mapping_order_cpu));
        void *q_vlabels_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&q_vlabels_cpu, q_vlabels, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&q_vlabels_cpu, &q_vlabels, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(q_vlabels_cpu));
        void *q_elabels_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&q_elabels_cpu, q_elabels, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&q_elabels_cpu, &q_elabels, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(q_elabels_cpu));
        void *q_starts_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&q_starts_cpu, q_starts, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&q_starts_cpu, &q_starts, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(q_starts_cpu));
        void *q_edges_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&q_edges_cpu, q_edges, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&q_edges_cpu, &q_edges, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(q_edges_cpu));
        void *g_vlabels_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&g_vlabels_cpu, g_vlabels, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&g_vlabels_cpu, &g_vlabels, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(g_vlabels_cpu));
        void *g_elabels_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&g_elabels_cpu, g_elabels, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&g_elabels_cpu, &g_elabels, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(g_elabels_cpu));
        void *g_starts_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&g_starts_cpu, g_starts, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&g_starts_cpu, &g_starts, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(g_starts_cpu));
        void *g_edges_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&g_edges_cpu, g_edges, sizeof(void *)));
        // HANDLE_RESULT(cudaMemcpy(&g_edges_cpu, &g_edges, sizeof(void *), cudaMemcpyDefault));
        HANDLE_RESULT(cudaFree(g_edges_cpu));
        void *sibling_pool_cpu;
        HANDLE_RESULT(cudaMemcpyFromSymbol(&sibling_pool_cpu, sibling_pool, sizeof(void *)));
        HANDLE_RESULT(cudaFree(sibling_pool_cpu));
    }

    states_pool_destroy(states_pool);
    lists_destroy(Ss, BLOCKS);
    heaps_destroy(Q, k);
    HANDLE_RESULT(cudaDeviceSynchronize());

    long long upper_bound_cpu;
    HANDLE_RESULT(cudaMemcpyFromSymbol(&upper_bound_cpu, upper_bound, sizeof(long long)));
    // HANDLE_RESULT(cudaMemcpy(&upper_bound_cpu, &upper_bound, sizeof(long long), cudaMemcpyDefault));
    return upper_bound_cpu;
}

__global__ void init_heap(heap **Q, state *states_pool) {
    heap_insert(Q[0], state_create(0, 0, 0, NULL, states_pool));
    atomicAdd(&total_Q_size, 1);
}
__device__ int processed = 0;
__device__ int steps = 0;
__device__ int heaps_min_before;

__global__ void clear_list(list *S) {
    list_clear(S);
}

__device__ void compute_mapped_cost(state *now) {
    int cost = 0;
    if (now->prev != NULL) {
        cost = now->prev->mapped_cost;
    }
    // check if the vertex is already mapped
    u8 vis_g[MAX_N], vis_q[MAX_N], rev_map[MAX_N];
    for (int i = 0; i < g_n; i++) {
        vis_g[i] = 0;
    }
    for (int i = 0; i < q_n; i++) {
        vis_q[i] = 0;
    }
    for (state *prev = now->prev; prev != NULL; prev = prev->prev) {
        vis_g[prev->image] = 1;
        rev_map[prev->image] = mapping_order[prev->level];
    }
    for (int i = 0; i < now->level; i++) {
        vis_q[mapping_order[i]] = 1;
    }

    int u = mapping_order[now->level];
    int v = now->image;
    // relabel this vertex?
    if (q_vlabels[u] != g_vlabels[v]) {
        cost += 1;
    }
    for (int j = q_starts[u]; j < q_starts[u + 1]; j++) {
        // delete this edge?
        if (vis_q[q_edges[j]]) cost += 1;
    }
    for (int j = g_starts[v]; j < g_starts[v + 1]; j++) {
        // relabel this edge?
        if (vis_g[g_edges[j]]) {
            cost += 1;
            int edge_label = q_matrix[u * q_n + rev_map[g_edges[j]]];
            if (edge_label == g_elabels[j]) {
                // identical edge, so revert both costs
                cost -= 2;
            } else if (edge_label < num_elabels) {
                // edge exists with different label
                cost += 1;
            }
        }
    }
    now->mapped_cost = cost;
}

__global__ void fill_list(int k, heap **Q, list *S, state *states_pool) {
    int id = calculate_id();
    if (id == 0)steps++;
    for (int i_ = id; i_ < k; i_ += blockDim.x * gridDim.x) {
        if (Q[i_]->size == 0) continue;
        state *q = heap_extract(Q[i_]);
        // printf("extracted %d\n", q->level);
        atomicSub(&total_Q_size, 1);
        // if (cuda_str_eq(q->node, t)) {
        //     if (m == NULL || f(q, t, h) < f(m, t, h)) {
        //         m = q;
        //     }
        //     continue;
        // }

        if (q->f > upper_bound) {
            continue;
        }

        state *sibling = state_create(0, 0, q->level, q->prev, states_pool);
        sibling->cs_cnt = 0;
        sibling->siblings = q->siblings;
        sibling->siblings_n = q->siblings_n;
        q->siblings = NULL;
        q->siblings_n = 0;
        short &tn = sibling->siblings_n;
        if (tn == 0) {
            sibling->image = g_n;
        } else {
            sibling->image = sibling->siblings[tn - 1].image;
            sibling->f = sibling->siblings[tn - 1].weight;
            tn -= 1;
            compute_mapped_cost(sibling);
        }

        if (sibling->image < g_n && sibling->f < upper_bound) {
            list_insert(S, sibling);
            if (q->prev != NULL) {
                q->prev->cs_cnt += 1;
            }
        } else {
            q->cs_cnt -= 1;
        }

        if (q->level == g_n - 1) {
            // reached the end of the query, extend it to full mapping
            u8 vis_q[MAX_N];
            for (int i = 0; i < q_n; i++) {
                vis_q[i] = 0;
            }

            int cost = q->mapped_cost;
            if (q_n < g_n) {
                for (state *prev = q; prev->prev != NULL; prev = prev->prev) {
                    vis_q[prev->image] = 1;
                }
                for (int i = 0; i < g_n; i++) {
                    if (!vis_q[i]) {
                        for (int j = g_starts[i]; j < g_starts[i + 1]; j++) {
                            if (vis_q[g_edges[j]]) {
                                cost += 1;
                            }
                        }
                        vis_q[i] = 1;
                    }
                }
                cost += g_n - q_n;
            }

            // todo: dump mapping
            if (cost < upper_bound) {
                printf("new upper bound: %d\n", cost);
            }
            atomicMin(&upper_bound, cost);

            continue;
        }

        // expand(q->node, my_expand_buf);
        // for (int j = 0; my_expand_buf[j][0] != '\0'; j++) {
        //     int delta = states_delta(q->node, my_expand_buf[j]);
        //     state *new_state = state_create(my_expand_buf[j], -1, q->g + delta, q, states_pool, nodes_pool, state_len);
        //     if (new_state == NULL) return;
        //     list_insert(S, new_state);
        // }

        int level = 0;
        int mapped_cost = 0;
        if (q->prev != NULL) {
            // not empty root node
            level = q->level + 1;
            mapped_cost = q->mapped_cost;
        }

        u8 vis_g[MAX_N], vis_q[MAX_N];
        for (int i = 0; i < g_n; i++) {
            vis_g[i] = 0;
        }
        for (int i = 0; i < q_n; i++) {
            vis_q[i] = 0;
        }
        for (state *prev = q; prev->prev != NULL; prev = prev->prev) {
            vis_g[prev->image] = 1;
        }
        int num_candidates = 0;
        int candidates[MAX_N];
        for (int i = 0; i < g_n; i++) {
            if (!vis_g[i]) {
                candidates[num_candidates++] = i;
            }
        }

        int vlabels_map[MAX_N];
        int elabels_map[MAX_N * 4];
        int elabels_matrix[MAX_N * MAX_N];
        for (int i = 0; i < num_vlabels; i++) {
            vlabels_map[i] = 0;
        }
        for (int i = 0; i < num_elabels; i++) {
            elabels_map[i] = 0;
        }
        for (int i = 0; i < q_n * num_elabels; i++) {
            elabels_matrix[i] = 0;
        }

        int size_q[MAX_N];
        int size_g[MAX_N];
        for (int i = 0; i < q_n; i++) {
            size_q[i] = 0;
        }
        for (int i = 0; i < g_n; i++) {
            size_g[i] = 0;
        }

        int u = mapping_order[level];
        int u_cross_cnt = 0;
        int u_inner_cnt = 0;
        for (int i = 0; i <= level; i++) {
            vis_q[mapping_order[i]] = 1;
        }
        for (int i = 0; i <= level; i++) {
            int v = mapping_order[i];
            for (int j = q_starts[v]; j < q_starts[v + 1]; j++) {
                if (!vis_q[q_edges[j]]) {
                    size_q[v] += 1;
                    elabels_matrix[v * num_elabels + q_elabels[j]] -= 1;
                }
            }
        }
        for (int i = q_starts[u]; i < q_starts[u + 1]; i++) {
            if (vis_q[q_edges[i]]) {
                u_cross_cnt += 1;
            }
        }
        for (int i = level + 1; i < q_n; i++) {
            int v = mapping_order[i];
            for (int j = q_starts[v]; j < q_starts[v + 1]; j++) {
                int uprime = q_edges[j];
                // only calculate once
                if (!vis_q[uprime] && v < uprime) {
                    u_inner_cnt += 1;
                    elabels_map[q_elabels[j]] -= 1;
                }
            }
        }
        for (int i = level + 1; i < q_n; i++) {
            vlabels_map[q_vlabels[mapping_order[i]]] -= 1;
        }

        u8 rev_map[MAX_N];
        for (int i = 0; i < g_n; i++) {
            rev_map[i] = g_n;
        }
        for (state *prev = q; prev->prev != NULL; prev = prev->prev) {
            int v = prev->image;
            vis_g[v] = 1;
            rev_map[v] = mapping_order[prev->level];
        }

        int ged_common = 0;
        for (state *st = q; st->prev != NULL; st = st->prev) {
            int v = st->image;
            int common = 0;
            size_g[v] = 0;
            for (int j = g_starts[v]; j < g_starts[v + 1]; j++) {
                int u = g_edges[j];
                if (!vis_g[u]) {
                    int &cnt = elabels_matrix[mapping_order[st->level] * num_elabels + g_elabels[j]];
                    if (cnt < 0) {
                        common += 1;
                    }
                    cnt += 1;
                    size_g[v] += 1;
                }
            }
            int t_ged = size_q[mapping_order[st->level]];
            if (size_g[v] > t_ged) {
                t_ged = size_g[v];
            }
            ged_common += t_ged - common;
        }

        int vl_common = 0, el_common = 0, v_total_cnt = 0;
        for (int i = 0; i < num_candidates; i++) {
            int v = candidates[i];
            if (vlabels_map[g_vlabels[v]] < 0) {
                vl_common += 1;
            }
            vlabels_map[g_vlabels[v]] += 1;
            for (int j = g_starts[v]; j < g_starts[v + 1]; j++) {
                int vprime = g_edges[j];
                if (!vis_g[vprime] && v < vprime) {
                    int &cnt = elabels_map[g_elabels[j]];
                    if (cnt < 0) {
                        el_common += 1;
                    }
                    cnt += 1;
                    v_total_cnt += 1;
                }
            }
        }

        int vl_ged = q_n - level - 1;
        if (g_n > q_n) {
            vl_ged = g_n - level - 1;
        }
        vl_ged -= vl_common;

        struct sibling children[MAX_N];
        for (int i = 0; i < num_candidates; i++) {
            int v = candidates[i];
            int lb = vl_ged + ged_common;
            if (vlabels_map[g_vlabels[v]] <= 0) {
                lb += 1;
            }

            int cross_ged = u_cross_cnt;
            int inner_ged = v_total_cnt;
            if (g_vlabels[v] != q_vlabels[u]) {
                cross_ged += 1;
            }
            int common = 0, t_size = 0;
            for (int j = g_starts[v]; j < g_starts[v + 1]; j++) {
                int vprime = g_edges[j];
                if (vis_g[vprime]) {
                    cross_ged += 1;
                    if (q_matrix[u * q_n + rev_map[vprime]] == g_elabels[j]) {
                        cross_ged -= 2;
                    } else if (q_matrix[u * q_n + rev_map[vprime]] < num_elabels) {
                        cross_ged -= 1;
                    }

                    if (elabels_matrix[rev_map[vprime] * num_elabels + g_elabels[j]] < 0) {
                        lb += 1;
                    }
                    if (size_g[vprime] > size_q[rev_map[vprime]]) {
                        lb -= 1;
                    }
                    continue;
                }

                inner_ged -= 1;
                if (elabels_map[g_elabels[j]] <= 0) {
                    el_common -= 1;
                }
                elabels_map[g_elabels[j]] -= 1;

                t_size += 1;
                if (elabels_matrix[u * num_elabels + g_elabels[j]] < 0) {
                    common += 1;
                }
                elabels_matrix[u * num_elabels + g_elabels[j]] += 1;
            }
            if (size_q[u] > t_size) {
                t_size = size_q[u];
            }
            lb += t_size - common;

            if (u_inner_cnt > inner_ged) {
                inner_ged = u_inner_cnt;
            }
            inner_ged -= el_common;
            // TODO: 不一致: children[i-pre_siblings].first = - mapped_cost - cross_ged - inner_ged - lb;
            children[i].weight = mapped_cost + cross_ged + inner_ged + lb;
            assert(mapped_cost >= 0);
            assert(cross_ged >= 0);
            assert(inner_ged >= 0);
            assert(lb >= 0);
            assert(children[i].weight >= 0);
            // TODO: 不一致: children[i-pre_siblings].second = v;
            children[i].image = v;

            for (int j = g_starts[v]; j < g_starts[v + 1]; j++) {
                if(!vis_g[g_edges[j]]) {
                    if (elabels_map[g_elabels[j]] < 0) {
                        el_common += 1;
                    }
                    elabels_map[g_elabels[j]] += 1;
                    elabels_matrix[u * num_elabels + g_elabels[j]] -= 1;
                }
            }
        }
        sort_siblings(children, children + num_candidates);
        int best = num_candidates - 1;

        int cross_ged = u_cross_cnt;
        for (int j = g_starts[children[best].image]; j < g_starts[children[best].image + 1]; j++) {
            int vprime = g_edges[j];
            if (vis_g[vprime]) {
                cross_ged += 1;
                if (q_matrix[u * q_n + rev_map[vprime]] == g_elabels[j]) {
                    cross_ged -= 2;
                } else if (q_matrix[u * q_n + rev_map[vprime]] < num_elabels) {
                    cross_ged -= 1;
                }
            }
        }
        int child_mc = mapped_cost + cross_ged;
        if (q_vlabels[u] != g_vlabels[children[best].image]) {
            child_mc += 1;
        }

        state *new_state = state_create(children[best].weight, child_mc, level, q, states_pool);
        if (new_state == NULL) return;
        new_state->image = children[best].image;

        new_state->siblings = alloc_siblings(num_candidates - 1);
        new_state->siblings_n = num_candidates - 1;
        for (int i = 0; i < num_candidates - 1; i++) {
            new_state->siblings[i].weight = children[i].weight;
            new_state->siblings[i].image = children[i].image;
        }

        if (new_state->image < g_n && new_state->f < upper_bound) {
            list_insert(S, new_state);
        } else {
            q->cs_cnt -= 1;
        }
    }
    
}

__global__ void push_to_queues(int k, heap **Q, list *S, int off) {
    for (int i = threadIdx.x; i < S->length; i += blockDim.x) {
        state *t1 = list_get(S, i);
        if (t1 != NULL) {
            heap_insert(Q[(i + off) % k], t1);
            atomicAdd(&processed, 1);
            atomicAdd(&total_Q_size, 1);
        }
        __syncthreads();
    }
}

__device__ int f(const state *x) {
    return x->f;
}

void states_pool_create(state **states) {
    HANDLE_RESULT(cudaMalloc(states, STATES * sizeof(state)));
    HANDLE_RESULT(cudaMemset(*states, 0, STATES * sizeof(state)));
}

void states_pool_destroy(state *states_pool) {
    HANDLE_RESULT(cudaFree(states_pool));
}

__device__ int used_states = 0;
__device__ state *state_create(int lb, int mc, int level, state *prev, state *states_pool) {
    int index = atomicAdd(&used_states, 1);
    if (index >= STATES) {
        out_of_memory = 1;
        return NULL;
    }
    state *result = &(states_pool[index]);
    result->f = lb;
    result->mapped_cost = mc;
    result->level = level;
    result->prev = prev;
    return result;
}

__device__ int calculate_id() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ inline int my_min(int a, int b) {
    return a < b ? a : b;
}

__device__ void sort_siblings(sibling *begin, sibling *end) {
    // [begin, end)
    // sort by weight, from high to low
    sibling buffer[MAX_N];
    int length = end - begin;
    for (int stride = 1; stride < length; stride *= 2) {
        for (int i = 0; i < length; i += 2 * stride) {
            int left = i;
            int right = my_min(i + stride, length);
            int left_end = right;
            int right_end = my_min(i + 2 * stride, length);
            int k = 0;
            while (left < left_end && right < right_end) {
                if (begin[left].weight > begin[right].weight) {
                    buffer[k++] = begin[left++];
                } else {
                    buffer[k++] = begin[right++];
                }
            }
            while (left < left_end) {
                buffer[k++] = begin[left++];
            }
            while (right < right_end) {
                buffer[k++] = begin[right++];
            }
            for (int j = 0; j < k; j++) {
                begin[i + j] = buffer[j];
            }
        }
    }
}

__device__ sibling *alloc_siblings(int n) {
    int end = atomicAdd(&used_siblings, n);
    if (end >= MAX_N) {
        out_of_memory = 1;
        return NULL;
    }
    return &(sibling_pool[end - n]);
}