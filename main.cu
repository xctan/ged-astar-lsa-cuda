#include <stdio.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <getopt.h>
#include "common.h"
#include "ged.h"

int threads_per_block = 128;
int blocks = 16;
int heap_length = 16 * 1024;

static struct option long_options[] = {
    {"threads-per-block", required_argument, NULL, 't'},
    {"blocks", required_argument, NULL, 'b'},
    {"heap-length", required_argument, NULL, 'q'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0},
};

int main(int argc, char *argv[]) {
    while (true) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "ht:b:q:", long_options, &option_index);
        if (c == -1) break;
        
        switch (c) {
        case 't':
            threads_per_block = atoi(optarg);
            break;
        case 'b':
            blocks = atoi(optarg);
            break;
        case 'q':
            heap_length = atoi(optarg);
            break;
        case 'h':
            printf("Usage: %s [options] [file]\n", argv[0]);
            printf("Options:\n");
            printf("  -t, --threads-per-block=NUM\n");
            printf("  -b, --blocks=NUM\n");
            printf("  -q, --heap-length=NUM\n");
            printf("  -h, --help\n");
            exit(0);
            break;
        default:
            printf("Unknown option: %c\n", c);
            exit(1);
        }
    }

    const char *filename = "data/temp.txt";
    if (optind < argc) filename = argv[optind];

    std::map<std::string, int> v_label_map, e_label_map;
    std::vector<Graph> graphs = 
        read_graph_from_file(filename, v_label_map, e_label_map);
    printf("number of graphs: %ld\n", graphs.size());
    printf("number of vertex labels: %ld\n", v_label_map.size());
    printf("number of edge labels: %ld\n", e_label_map.size());

    int n = graphs.size();
    int ged = ged_astar_gpu(graphs[0], graphs[1]);

    printf("ged: %d\n", ged);

    return 0;
}