#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include "graph.h"
#include "common.h"

/// @brief Reassign a label to an integer id.
/// @param map 
/// @param label 
/// @return 
auto remap_label(std::map<std::string, int> &map, const char *label) -> int
{
    auto it = map.find(label);
    if (it == map.end()) {
        int id = map.size();
        map[label] = id;
        return id;
    } else {
        return it->second;
    }
}

auto construct_graph(std::vector<Vertex> vertices, std::vector<Edge> edges) -> Graph
{
    std::sort(vertices.begin(), vertices.end(), [](const Vertex &a, const Vertex &b) {
        return a.id < b.id;
    });
    std::sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b) {
        return a.src < b.src || (a.src == b.src && a.dst < b.dst);
    });

    // ensure vertices are numbered from 0 to n - 1 contiguously
    int n = vertices.size();
    for (int i = 0; i < n; i++) {
        my_assert(vertices[i].id == i && "Error: vertices are not numbered from 0 to n - 1 contiguously");
    }

    // ensure src and dst are valid in edges
    int m = edges.size();
    for (int i = 0; i < m; i++) {
        my_assert(edges[i].src >= 0 && edges[i].src < n && "Error: invalid src in edges");
        my_assert(edges[i].dst >= 0 && edges[i].dst < n && "Error: invalid dst in edges");
        if (i > 0) {
            my_assert((edges[i].src != edges[i - 1].src || edges[i].dst != edges[i - 1].dst) &&
                      "Error: duplicate edges");
        }
    }

    int *vertex_labels = new int[n];
    int *edge_from_sep = new int[n + 1];
    int *edge_to = new int[m];
    int *edge_labels = new int[m];

    std::fill(vertex_labels, vertex_labels + n, 0);
    for (int i = 0; i < n; i++) {
        vertex_labels[i] = vertices[i].v_label_id;
    }
    // calculate edge_from_sep[]
    // first count edges of each vertex
    std::fill(edge_from_sep, edge_from_sep + n + 1, 0);
    for (int i = 0; i < m; i++) {
        edge_from_sep[edges[i].src + 1]++;
        edge_to[i] = edges[i].dst;
        edge_labels[i] = edges[i].e_label_id;
    }
    // then compute prefix sum as index range of edges of each vertex
    for (int i = 1; i <= n; i++) {
        edge_from_sep[i] += edge_from_sep[i - 1];
    }

    return {n, m, vertex_labels, edge_from_sep, edge_to, edge_labels};
}

auto read_graph_from_file(const char *filename,
                          std::map<std::string, int> &v_label_map,
                          std::map<std::string, int> &e_label_map)
    -> std::vector<Graph>
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        panic("Error: could not open file %s\n", filename);
    }
    
    char linebuf[1024], labelbuf[128];
    int a, b;
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Graph> graphs;

    enum {
        WAIT_FOR_BEGIN,
        READING,
    } state = WAIT_FOR_BEGIN;
    auto finalize_reading = [&] {
        if (state == READING) {
            graphs.push_back(construct_graph(std::move(vertices), std::move(edges)));
        }
    };
    while (fgets(linebuf, sizeof(linebuf), fp) != NULL) {
        if (linebuf[0] == '#') {
            continue;
        }
        if (state == WAIT_FOR_BEGIN) {
            if (linebuf[0] == 't') {
                state = READING;
            }
        } else if (state == READING) {
            switch (linebuf[0]) {
            case 'v':
                sscanf(linebuf + 1, "%d %128s", &a, labelbuf);
                vertices.push_back({a, remap_label(v_label_map, labelbuf)});
                break;
            case 'e':
                sscanf(linebuf + 1, "%d %d %128s", &a, &b, labelbuf);
                edges.push_back({a, b, remap_label(e_label_map, labelbuf)});
                edges.push_back({b, a, remap_label(e_label_map, labelbuf)});
                break;
            case 't':
                finalize_reading();
                break;
            default:
                panic("Error: invalid line: %s\n", linebuf);
            }
        }
    }
    finalize_reading();

    fclose(fp);
    return graphs;
}
