#include <torch/extension.h>
#include <iostream>
#include <vector>


using namespace std;


/* 
    Below is the C++ implementation of Kruskal's algorithm,
    equivalent to the Pytorch implementation.
*/
std::vector<torch::Tensor> get_root(torch::Tensor parents, torch::Tensor node){
    std::vector<torch::Tensor> path = {node};
    torch::Tensor root = parents[node];
    while (root.item<int>() != path.back().item<int>()){
        path.push_back(root);
        root = parents[root];
    }

    torch::Tensor parents_new = parents.narrow_copy(0, 0, parents.size(0));
    // Compress the path and return.
    for(auto const& value: path) {
        parents_new[value] = root;
    }
    return {parents_new, root};
}

torch::Tensor get_tree(torch::Tensor edges, int n) {
    torch::Tensor adj_matrix = torch::zeros({edges.size(0), n - 1, n});
    for (int sample_idx=0; sample_idx < edges.size(0); sample_idx++){
        // Initialize weights and edges.
        torch::Tensor weights = torch::ones({n});
        torch::Tensor parents = torch::arange(0, n);

        int num_selected_edges = 0;
        for (int edge_idx=0; edge_idx < edges.size(1); edge_idx++){
            torch::Tensor i = edges[sample_idx][edge_idx][0];
            torch::Tensor j = edges[sample_idx][edge_idx][1];
            
            torch::Tensor root_i = i;
            std::vector<torch::Tensor> parents_and_root_i = get_root(parents, i);
            parents = parents_and_root_i[0];
            root_i = parents_and_root_i[1];
            torch::Tensor root_j = j;
            std::vector<torch::Tensor> parents_and_root_j = get_root(parents, j);
            parents = parents_and_root_j[0];
            root_j = parents_and_root_j[1];

            // Combine two forests if i and j are not in the same forest.
            if (root_i.item<int>() != root_j.item<int>()){
                torch::Tensor heavier = (
                    (weights[root_i], root_i).max(), 
                    (weights[root_j], root_j).max()).max();

                if (root_i.item<int>() != heavier.item<int>()){
                    weights[heavier] = weights[heavier] + weights[root_i];
                    parents[root_i] = heavier;
                }
                if (root_j.item<int>() != heavier.item<int>()){
                    weights[heavier] = weights[heavier] + weights[root_j];
                    parents[root_j] = heavier;
                }
                // Update adjacency matrix.
                adj_matrix[sample_idx][i][j] = 1;
                adj_matrix[sample_idx][j - 1][i] = 1;

                num_selected_edges++;
                if (num_selected_edges == (n - 1)){
                    break;
                }
            }
        }
    }
    return adj_matrix;
}


/* 
    Below is another C++ implementation of Kruskal's algorithm,
    copied from https://github.com/condaatje/Kruskal.
*/
class GEdge {
public:
    GEdge(int v1, int v2, double weight) {
        this->v1 = v1;
        this->v2 = v2;
        this->weight = weight;
    }
    int v1;
    int v2;
    double weight;
};


bool edgeCompare (GEdge a, GEdge b);

class GNode {
public:
    GNode * parent;
    int rank;
    int value;
};

class Union_Find {
private:
    vector<GNode *> sets; // this makes cleanup and testing much easier.
    GNode * link(GNode * a, GNode * b);
    
public:
    Union_Find(int size);
    Union_Find();
    void makeset(int x);
    void onion(int x, int y);
    GNode * find(int x);
    void clean();
    vector<GNode *> raw();
};


bool edgeCompare (GEdge a, GEdge b) { return (a.weight > b.weight); }

GNode * Union_Find::link(GNode * a, GNode * b) {
    // put the smaller rank tree as a child of the bigger rank tree.
    // otherwise (equal rank), put second element as parent.
    if (a->rank > b->rank) {
        // swap pointers
        GNode * temp_ptr = b;
        b = a;
        a = temp_ptr;
    }
    if (a->rank == b->rank) {
        // update the rank of the new parrent
        b->rank = b->rank + 1;
    }
    
    // a is child of b
    a->parent = b;
    return b;
}

Union_Find::Union_Find(int size) {
    // optimized init.
    sets.resize(size);
}

Union_Find::Union_Find() {}

void Union_Find::makeset(int x) {
    // takes in a vertex. creates a set out of it solo.
    
    GNode * n = new GNode();
    n->value = x;
    n->rank = 0;
    n->parent = n;
    
    if (sets.size() <= x) {
        sets.resize(x + 1); // +1 handles 0 index, but watch out for other issues.
        // Best to initialize with a suggested size.
    }
    sets[x] = n;
}

// "union" is taken
void Union_Find::onion(int x, int y) {
    // replace two sets containing x and y with their union.
    this->link(this->find(x), this->find(y));
}

GNode * Union_Find::find(int x) {
    GNode * n = sets[x];
    
    if (n->parent->value != n->value) {
        // walk the node up the tree (flattens as it finds)
        n->parent = find(n->parent->value);
    }
    
    return n->parent;
}

void Union_Find::clean() {
    // Normally I would just make a destructor,
    // but scoping is strange with iterative deepening.
    
    for(int i = 0; i < sets.size(); i++) {
        free(sets[i]);
    }
    sets.clear();
}

vector<GNode *> Union_Find::raw() {
    return sets;
};

torch::Tensor kruskals(torch::Tensor weights_and_edges, int n) {
    int batch_size = weights_and_edges.size(0);
    torch::Tensor adj_matrix = torch::zeros({batch_size, n - 1, n});
    for (int sample_idx=0; sample_idx < batch_size; sample_idx++) {
        Union_Find uf;
        
        vector<GEdge> edges;
        
        for(int idx = 0; idx < (n * (n - 1) / 2); idx++) {
            double w = weights_and_edges[sample_idx][idx][0].item<double>();
            int i = weights_and_edges[sample_idx][idx][1].item<int>();
            int j = weights_and_edges[sample_idx][idx][2].item<int>();

            GEdge e(i, j, w);
            edges.push_back(e);

            // Insert vertices into union find.
            if (idx < n) { uf.makeset(idx); }
        }
        
        sort(edges.begin(), edges.end(), edgeCompare);

        int mst_count = 0;
        for(int i = 0; i < edges.size(); i++) {
            GEdge e = edges[i];
            if (uf.find(e.v1) != uf.find(e.v2)) {
                uf.onion(e.v1, e.v2);
                mst_count = mst_count + 1;

                // Update adjacency matrix.
                adj_matrix[sample_idx][e.v1][e.v2] = 1;
                adj_matrix[sample_idx][e.v2 - 1][e.v1] = 1;

                if (mst_count >= (n - 1)) { // |V| - 1 edges
                    break;
                }
            }
        }
        
        uf.clean();
    }
    return adj_matrix;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_tree", &get_tree, "Get Tree");
  m.def("get_root", &get_root, "Get Root");
  m.def("kruskals", &kruskals, "Kruskals");
}