#ifndef __GRAPH_H__
#define __GRAPH_H__

struct Graph {
	// number of vertices
    int num_vertices;
	// number of edges
    int num_edges;
	// array of vertex labels
    int *vertex_labels;
	// separators of edges of different source vertices
    int *edge_from_sep;
	// destinations of edges, sorted by source vertices
    int *edge_to;
	// labels of edges
    int *edge_labels;

    // destructor, maybe?
};

struct Vertex {
    int id;
    int v_label_id;
};

struct Edge {
    int src;
    int dst;
    int e_label_id;
};

auto read_graph_from_file(const char *filename,
                          std::map<std::string, int> &v_label_map,
                          std::map<std::string, int> &e_label_map)
    -> std::vector<Graph>;

#endif // __GRAPH_H__