#include<iostream>
#include<string>
#include<cstring>
#include<vector>
#include<algorithm>
#include<deque>
#include "node.h"
// #include <omp.h>
#include<unordered_map>
#include<cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;

#define MAX 1000000000.0
#define INTERNAL_NODE_ID_START 1000

template <typename T>
void print_2d_vec(const vector<vector<T>>& M) {
    for(int i=0;i<M.size();i++){
        for(int j=0;j<M[0].size();j++){
            cout << M[i][j] << " ";
        }
        cout << endl;
    }
}

template <typename T>
void print_1d_vec(const vector<T>& M) {
    for(int i=0;i<M.size();i++){
        cout << M[i] << " ";
    }
    cout << endl;
}


template <typename T>
void print_2d_arr(T** arr, int m, int n) {
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            cout << arr[i][j] << " ";
        }
        cout << endl;
    }
}

template <typename T>
void memset_2d_arr(T** arr, int m, int n, T val) {
    for(int i=0;i<m;i++) 
        for(int j=0;j<n;j++)
            arr[i][j] = val;
}

template <typename T>
void memset_3d_arr(T*** arr, int m, int n, int l, T val) {
    for(int i=0;i<m;i++) 
        for(int j=0;j<n;j++)
            for(int k=0;k<l;k++)
                arr[i][j][k] = val;
}

template <typename T>
void memset_1d_arr(T* arr, int m, T val) {
    for(int i=0;i<m;i++) arr[i] = val;
}



void getCommonParentMatrix(Node* tree, vector<vector<int>>& matrix) {
    if(tree->isLeaf()){
        matrix[tree->id][tree->id] = tree->tmp_id;
        return;
    }
    for(int i=0;i<tree->children.size();i++){
        for(int j=i+1;j<tree->children.size();j++){
            for(auto li: tree->children[i]->leaves){
                for(auto lj: tree->children[j]->leaves){
                    matrix[li->id][lj->id] = tree->id;
                    matrix[lj->id][li->id] = tree->id;
                }
            }
        }
    }
    for(auto c: tree->children) 
        getCommonParentMatrix(c, matrix);
}


void setId(Node* tree, unordered_map<string, int>& name2id) {
    int inter_start_id = INTERNAL_NODE_ID_START;
    int start_id = 0;
    deque<Node*> que;
    que.push_back(tree);
    while(que.size()>0){
        Node* p = que[0];
        que.pop_front();
        if(p->isLeaf()){
            if(name2id.count(p->name)!=0)
                p->id = name2id[p->name];
            else {
                // level kolo
                p->id = start_id;
                start_id++;
            }
        } else {
            p->id = inter_start_id;
            inter_start_id++;
            que.insert(que.end(), p->children.begin(), p->children.end());
        }
    }
}

void printTree(Node* tree) {
    if(tree->isLeaf()){
        cout << "Leaf: " << tree->id << " " << tree->name << endl;
    } else {
        cout << endl;
        cout << "Inter: " << tree->id << " " << tree->name << endl;
        cout << "leaves: " << tree->leaves.size() << endl;
        for(auto p_l:tree->leaves) cout << p_l->name << " ";
        cout << endl;
        for(auto c: tree->children) printTree(c);
    }
}

void setLeaves(Node* tree){
    if(tree->isLeaf()){
        tree->leaves.push_back(tree);
        tree->leavesId.push_back(tree->id);
        return;
    }
    for(auto child: tree->children){
        setLeaves(child);
        for(auto l: child->leaves) {
            tree->leaves.push_back(l);
            tree->leavesId.push_back(l->id);
        }
    }
}

Node* hierarchyToTree(unordered_map<string, vector<string>>& hierarchy) {
    unordered_map<string, Node*> nodes;
    for(auto iter=hierarchy.begin(); iter!=hierarchy.end(); iter++) {
        string name = iter->first;
        vector<string> children = iter->second;
        Node* par;
        if(nodes.count(name)==0){
            par = new Node(name);
            nodes[name] = par;
        } else{
            par = nodes[name];
        }
        for(auto c_name:children) {
            Node* child;
            if(nodes.count(c_name)==0){
                child = new Node(c_name);
                nodes[c_name] = child; 
            } else {
                child = nodes[c_name];
            }
            par->children.push_back(child);
        }
    }
    return nodes["root"];
}

void printHierarchy(unordered_map<string, vector<string>> &hierarchy) {
    for(auto iter=hierarchy.begin(); iter!=hierarchy.end(); iter++){
        cout << iter->first << ": ";
        for(auto name: iter->second)
            cout << name << " ";
        cout << endl;
    }
}




Node initTree(int N) {
    Node root = Node(1000, "root");
    for(int i=0; i<3; i++){
        Node* tmp = new Node(2000+i, to_string(2000+i));
        // tmp->leaves.push_back(tmp);
        root.children.push_back(tmp);
        // root.leaves.push_back(tmp);
    }
    int idx = 0;
    for(auto c: root.children){
        for(int i=0;i<3;i++){
            Node* tmp = new Node(idx, to_string(idx));
            c->children.push_back(tmp);
            idx+=1;
        }
    }

    return root;
}

Node initOneTree(int N) {
    Node root = Node(1000, "root");
    for(int i=0; i<N; i++){
        Node* tmp = new Node(i, to_string(i));
        // tmp->leaves.push_back(tmp);
        root.children.push_back(tmp);
        // root.leaves.push_back(tmp);
    }
    return root;
}

void travel(vector<int> path, int cur, int n, vector<bool>& vis, vector<vector<int>>& res) {
    if(path.size()==n/2) {
        res.push_back(path);
        return;
    }
    if(cur==n) return;
    
    vis[cur] = true;
    path.push_back(cur);
    travel(path, cur + 1, n, vis, res);
    vis[cur] = false;
    path.pop_back();
    travel(path, cur+1, n, vis, res);
    
};



vector<pair<vector<int>, vector<int>>> getComb(int n) {
    vector<bool> vis(n);
    for(int i=0;i<n;i++) vis[i] = false;
    vector<vector<int>> res;
    int cur = 0;
    vector<int> path;
    travel(path, cur, n, vis, res);
    vector<pair<vector<int>, vector<int>>> combs;
    for(int i=0; i<res.size(); i++) {
        vector<int> left_idx = res[i];
        vector<int> right_idx;
        for(int j=0; j<n; j++) {
            if(find(left_idx.begin(), left_idx.end(), j)==left_idx.end()){
                right_idx.push_back(j);
            }
        }
        combs.push_back(make_pair(left_idx, right_idx));
    }
    if(n%2==0){
        combs.erase(combs.begin()+combs.size()/2, combs.end());
    }
    return combs;
}


class KOLO {
    public:
        Node* tree;
        vector<vector<double>> SM;

        vector<vector<int>> PM; // 记录树中任意两叶节点的最小公共祖先
        double** M;
        vector<vector<vector<int>>> I;
        double*** D;
        vector<vector<vector<vector<int>>>> DI;
        bool* vis;
        int K = 9; // K叉树
        int N = 100; // 叶节点数

        KOLO(Node* u_tree, const vector<vector<double>>& u_SM);
        ~KOLO();
        void initD();
        void clearD();
        void deleteD();

        void getOrdered();
        void opt(Node* v);
        void mem(vector<Node*>& x, int p_id); // 第二个参数为父节点id
        void compute(vector<Node*> L, vector<Node*> R);

        int encode(vector<int> ids);
        vector<int> decode(int id);
};

void KOLO::getOrdered(){
    opt(tree);
    // print_2d_arr(M, N, N);
}

KOLO::KOLO(Node* u_tree, const vector<vector<double>>& u_SM) {
    tree = u_tree;
    SM = u_SM;

    N = tree->leaves.size();
    K = tree->children.size(); // todo: max children_num of interval nodes

    initD();
    
    M = new double*[N];
    for(int i=0;i<N;i++){
        M[i] = new double[N];
    }

    for(int i=0;i<N;i++){
        PM.push_back(vector<int>(N));
    }

    getCommonParentMatrix(tree, PM);

    for(int i=0;i<N;i++){
        I.push_back(vector<vector<int>>());
        for(int j=0;j<N;j++) I[i].push_back(vector<int>());
    }

    // cout << "PM" << endl;
    // print_2d_vec(PM);

    // cout << "M" << endl;
    // print_2d_arr(M, N, N);
}

void KOLO::initD() {
    int dim = (int)pow(2.0, K);
    D = new double**[dim];
    for(int i=0;i<dim;i++){
        D[i] = new double*[N];
        for(int j=0;j<N;j++){
            D[i][j] = new double[N];
        }
    }
    memset_3d_arr(D, dim, N, N, MAX);

    vis = new bool[dim];
    memset_1d_arr(vis, dim, false);

    for(int i=0;i<dim;i++){
        DI.push_back(vector<vector<vector<int>>>());
        for(int j=0;j<N;j++){
            DI[i].push_back(vector<vector<int>>());
            for(int k=0;k<N;k++){
                DI[i][j].push_back(vector<int>());
            }
        }
    }
}

void KOLO::clearD(){
    int dim = (int)pow(2.0, K);
    memset_3d_arr(D, dim, N, N, MAX);
    memset_1d_arr(vis, dim, false);

    for(int i=0;i<dim;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                DI[i][j][k].clear();
            }
        }
    }
}

void KOLO::deleteD() {
    int dim = (int)pow(2.0, K);
    for(int i=0;i<dim;i++){
        for(int j=0;j<N;j++){
            delete[] D[i][j];
        }
        delete[] D[i];
    }
    delete[] D;
    delete[] vis;
}

KOLO::~KOLO() {
    for(int i=0; i<N; i++) {
        delete[] M[i];
    }
    delete[] M;
}

int KOLO::encode(vector<int> ids){
    int idx = 0;
    for(auto id: ids) idx |= 1<<id;
    return idx;
}

vector<int> KOLO::decode(int id){
    vector<int> ids;
    int idx = 0;
    while(id>0){
        if(id%2==1) ids.push_back(idx);
        id >>= 1;
        idx += 1;
    }
    return ids;
}

void KOLO::opt(Node* v){
    // cout << "id: " << v->id << endl;
    // print_1d_vec(v->id);
    if(v->isLeaf()){
        M[v->id][v->id] = 0;
        I[v->id][v->id].push_back(v->id);
        return;
    }

    for(auto child: v->children) opt(child);

    clearD();

    vector<int> v_tmp_id;
    for(int i=0;i<v->children.size();i++){
        v->children[i]->tmp_id = i;
        v_tmp_id.push_back(i);
    }
    
    mem(v->children, v->id);
    vector<int>& v_ids = v->leavesId;
    double** v_M = D[encode(v_tmp_id)];
    vector<vector<vector<int>>>* v_I = &DI[encode(v_tmp_id)]; 
    for(int i=0;i<v_ids.size();i++){
        for(int j=0;j<v_ids.size();j++){
            M[v_ids[i]][v_ids[j]] = v_M[v_ids[i]][v_ids[j]];
            I[v_ids[i]][v_ids[j]] = (*v_I)[v_ids[i]][v_ids[j]];
        }
    }

    // cout << encode(v_tmp_id) << endl;
    // print_1d_vec(v->getSortedLeavesId());
    // print_2d_arr(M, N, N);

    clearD();
}

void KOLO::mem(vector<Node*>& x, int pid){

    vector<int> x_tmp_id;
    for(auto n: x) x_tmp_id.push_back(n->tmp_id);
    int num_x_tmp_id = encode(x_tmp_id);


    if(x.size()==1) {
        for(auto i: x[0]->leavesId){
            for(auto j: x[0]->leavesId){
                D[num_x_tmp_id][i][j] = M[i][j];
                DI[num_x_tmp_id][i][j] = I[i][j];
            }
        }
        vis[num_x_tmp_id] = true;
        return;
    }

    if(vis[num_x_tmp_id]) return;

    // cout << "vec_id" <<  endl;
    // for(auto n:x) cout << n->id << " ";
    // cout << endl;
    // cout << "tmp_id" << endl;
    // print_1d_vec(x_tmp_id);
    // cout << "num" << endl;
    // cout << num_x_tmp_id;
    // cout << endl;

    double** best_m = D[num_x_tmp_id];
    vector<vector<vector<int>>>* best_i = &DI[num_x_tmp_id];

    vector<pair<vector<int>, vector<int>>> combs = getComb(x.size());

    double** tmp_il_val;
    tmp_il_val = new double*[N];
    for(int i=0;i<N;i++) tmp_il_val[i] = new double [N];
    int** tmp_h_id;
    tmp_h_id = new int*[N];
    for(int i=0;i<N;i++) tmp_h_id[i] = new int[N];

    for(int idx=0;idx<combs.size();idx++){
        vector<int> left_idx = combs[idx].first;
        vector<int> right_idx = combs[idx].second;
        vector<Node*> left_nodes, right_nodes;
        for(auto idx: left_idx) left_nodes.push_back(x[idx]);
        for(auto idx: right_idx) right_nodes.push_back(x[idx]);

        // double** 
        // double** 

        mem(left_nodes, pid);
        mem(right_nodes, pid);

        vector<int> tmp_id;
        int left_tmp_id, right_tmp_id;
        double** l_m;
        double** r_m;

        for(auto n: left_nodes) tmp_id.push_back(n->tmp_id);
        left_tmp_id = encode(tmp_id);
        tmp_id.clear();
        for(auto n: right_nodes) tmp_id.push_back(n->tmp_id);
        right_tmp_id = encode(tmp_id);


        vector<vector<vector<int>>>* l_i = &DI[left_tmp_id];
        vector<vector<vector<int>>>* r_i = &DI[right_tmp_id];

        l_m = D[left_tmp_id];
        r_m = D[right_tmp_id];
        // l_m = mem(left_nodes, pid);
        // r_m = mem(right_nodes, pid);

        memset_2d_arr(tmp_il_val, N, N, MAX);
        memset_2d_arr(tmp_h_id, N, N, -1);
        
        double min_il_val = MAX;
        int best_h_id = -1;

        vector<Node*> left_leaves;
        for(auto l: left_nodes) left_leaves.insert(left_leaves.end(), l->leaves.begin(), l->leaves.end());
        vector<Node*> right_leaves;
        for(auto r: right_nodes) right_leaves.insert(right_leaves.end(), r->leaves.begin(), r->leaves.end());

        for(auto l: right_leaves){
            for(auto i: left_leaves){
                min_il_val = MAX;
                best_h_id = -1;
                for(auto h: left_leaves){
                    if(PM[i->id][h->id]!=pid && left_nodes.size()!=1) continue;
                    if(min_il_val > l_m[i->id][h->id] + SM[h->id][l->id]){
                        min_il_val = l_m[i->id][h->id] + SM[h->id][l->id];
                        best_h_id = h->id;
                    }
                }
                tmp_il_val[i->id][l->id] = min_il_val;
                tmp_h_id[i->id][l->id] = best_h_id;
            }
        }

        double min_ij_val = MAX;
        int best_l_id = -1;
        for(auto i: left_leaves){
            for(auto j: right_leaves){
                min_ij_val = MAX;
                best_l_id = -1;
                for(auto l: right_leaves){
                    if(PM[j->id][l->id]!=pid && right_nodes.size()!=1)continue;
                    if(min_ij_val > tmp_il_val[i->id][l->id] + r_m[l->id][j->id]){
                        min_ij_val = tmp_il_val[i->id][l->id] + r_m[l->id][j->id];
                        best_l_id = l->id;
                    }
                }
                if(best_m[i->id][j->id] > min_ij_val){
                    best_m[i->id][j->id] = min_ij_val;
                    best_m[j->id][i->id] = min_ij_val;
                    vector<int> tmp;
                    int h_id = tmp_h_id[i->id][best_l_id];
                    tmp.insert(tmp.end(), (*l_i)[i->id][h_id].begin(), (*l_i)[i->id][h_id].end());
                    tmp.insert(tmp.end(), (*r_i)[best_l_id][j->id].begin(), (*r_i)[best_l_id][j->id].end());
                    (*best_i)[i->id][j->id] = tmp;
                    reverse(tmp.begin(), tmp.end());
                    (*best_i)[j->id][i->id] = tmp;
                }
            }
        }
    }
    // print_2d_arr(best_m, N, N);
    // cout << endl;
    vis[num_x_tmp_id] = true;
    // return best_m;
}

// int main() {
//     int N = 15;
//     Node tree = initTree(N);
//     setLeaves(&tree);

//     vector<vector<double>> SM;
//     for(int i=0;i<N;i++) SM.push_back(vector<double>(N));
//     for(int i=0;i<N;i++) for(int j=0;j<N;j++) SM[i][j] = 100;
//     for(int i=0;i<N-1;i++){
//         SM[i][i+1] = 10;
//         SM[i+1][i] = 10;
//     }
//     SM[0][N-1] = 10;
//     SM[N-1][0] = 10;
//     KOLO kolo = KOLO(&tree, SM);

//     kolo.getOrdered();
//     double minm = MAX;
//     int min_i, min_j;
//     for(int i=0;i<kolo.N;i++){
//         for(int j=0;j<kolo.N;j++){
//             if(minm>kolo.M[i][j]) {
//                 minm = kolo.M[i][j];
//                 min_i = i;
//                 min_j = j;
//             }
//         }
//     }
//     cout << minm << endl;
//     print_1d_vec(kolo.I[min_i][min_j]);

//     // vector<vector<double>> M;
//     // for(int i=0;i<kolo.N;i++){
//     //     M.push_back(vector<double>(kolo.N));
//     //     for(int j=0;j<kolo.N;j++){
//     //         M[i][j] = kolo.M[i][j];
//     //     }
//     // }
//     // print_2d_vec(M);
    
//     return 0;
// }

int main() {
    int N = 15;
    Node tree = initOneTree(N);
    setLeaves(&tree);

    vector<vector<double>> SM;
    for(int i=0;i<N;i++) SM.push_back(vector<double>(N));
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) SM[i][j] = 100;
    for(int i=0;i<N-1;i++){
        SM[i][i+1] = 10;
        SM[i+1][i] = 10;
    }
    SM[0][N-1] = 10;
    SM[N-1][0] = 10;
    KOLO kolo = KOLO(&tree, SM);

    kolo.getOrdered();
    double minm = MAX;
    int min_i, min_j;
    for(int i=0;i<kolo.N;i++){
        for(int j=0;j<kolo.N;j++){
            if(minm>kolo.M[i][j]) {
                minm = kolo.M[i][j];
                min_i = i;
                min_j = j;
            }
        }
    }
    cout << minm << endl;
    print_1d_vec(kolo.I[min_i][min_j]);

    // vector<vector<double>> M;
    // for(int i=0;i<kolo.N;i++){
    //     M.push_back(vector<double>(kolo.N));
    //     for(int j=0;j<kolo.N;j++){
    //         M[i][j] = kolo.M[i][j];
    //     }
    // }
    // print_2d_vec(M);
    
    return 0;
}


void print_map(unordered_map<string, int> m){
    for(auto iter=m.begin();iter!=m.end();iter++){
        cout << (*iter).first << " " << (*iter).second << endl;
    }
}

vector<int> kolo_reordering(vector<vector<double>> SM, unordered_map<string, int> name2id, unordered_map<string, vector<string>> hierarchy){
    // print_2d_vec(SM);
    // printHierarchy(hierarchy);
    // print_map(name2id);
    Node* tree = hierarchyToTree(hierarchy);
    // printTree(tree);
    setId(tree, name2id);
    // printTree(tree);
    setLeaves(tree);
    // printTree(tree);
    KOLO kolo = KOLO(tree, SM);

    kolo.getOrdered();
    double minm = MAX;
    int min_i, min_j;
    for(int i=0;i<kolo.N;i++){
        for(int j=0;j<kolo.N;j++){
            if(minm>kolo.M[i][j]) {
                minm = kolo.M[i][j];
                min_i = i;
                min_j = j;
            }
        }
    }
    // cout << minm << endl;
    // print_1d_vec(kolo.I[min_i][min_j]);

    vector<vector<double>> M;
    for(int i=0;i<kolo.N;i++){
        M.push_back(vector<double>(kolo.N));
        for(int j=0;j<kolo.N;j++){
            M[i][j] = kolo.M[i][j];
        }
    }
    
    return kolo.I[min_i][min_j];
}

PYBIND11_MODULE(reordering, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("kolo_reordering", &kolo_reordering);
    py::class_<Node>(m, "Node", py::dynamic_attr())
        .def(py::init<const string &>())
        .def(py::init<int, const string &>())
        .def_readwrite("id", &Node::id)
        .def_readwrite("name", &Node::name)
        .def_readwrite("children", &Node::children)
        .def_readwrite("leaves", &Node::leaves)
        .def("isLeaf", &Node::isLeaf)
        .def("getSortedLeavesId", &Node::getSortedLeavesId);
}