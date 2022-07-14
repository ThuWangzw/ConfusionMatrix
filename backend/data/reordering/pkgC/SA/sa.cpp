// #pragma GCC optimize(2)
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include<string>
#include <sstream>
#include <omp.h>
#include <map>
#include "node.h"
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

/****************************************************************/
/*
    This programme implement a simulated annealing for the
    quadratic assignment problem along the lines describes in
    the article D. T. Connoly, "An improved annealing scheme for
    the QAP", European Journal of Operational Research 46, 1990,
    93-100.

    Compiler : g++ or CC should work.

    Author : E. Taillard,
             EIVD, Route de Cheseaux 1, CH-1400 Yverdon, Switzerland

    Date : 16. 3. 98

    Format of data file : Example for problem nug5 :

5

0 1 1 2 3
1 0 2 1 2
1 2 0 1 2
2 1 1 0 1
3 2 2 1 0

0 5 2 4 1
5 0 3 0 2
2 3 0 0 0
4 0 0 0 5
1 2 0 5 0

   Additionnal parameters : Number of iterations, number of runs

*/

/********************************************************************/

#define INTERNAL_NODE_ID_START 2000
constexpr int N_MAX = 1005;
constexpr int N_LEVEL_MAX = 15;
typedef double mat[N_MAX][N_MAX];
int n;
mat a, b;
string path = "/data/fengyuan/ConfusionMatrix/backend/data/reordering/pkgC/data_test/";
map<string, int> name2id;
map<string, vector<string>> hierarchy;

double level_dis_mat[N_LEVEL_MAX][N_MAX][N_MAX];
double level_bar_mat[N_LEVEL_MAX][N_MAX][N_MAX];
int ids[N_MAX];
vector<double> avgs;
vector<double> stds;

void get_interval_nodes(Node* &node, vector<Node*> &intervals){
    if(node->children.empty()) return;
    intervals.push_back(node);
    for(int i=0;i<node->children.size();i++) get_interval_nodes(node->children[i], intervals);
}

double getAvg(vector<double> arr)
{
  double sum = 0;
  for (int i = 0; i < arr.size(); i++)
    sum += arr[i];
  return sum / arr.size();
}

double getStd(vector<double> arr)
{
  double avg = getAvg(arr);
  double sum = 0;
  for (int i = 0; i < arr.size(); i++)
  {
    sum += (arr[i] - avg) * (arr[i] - avg);
  }
  return sqrt(sum / (arr.size() - 1));
}

/* Generates an integer in [min, max] */
int generateRandomNumber(int min, int max)
{
  double rand_aux = ((double)rand() / ((double)(RAND_MAX) + (double)(1)));
  int r = (rand_aux * (double)(max - min + 1)) + min;
  return r;
}

/* Generates a real number in [0, 1] */
double generateDoubleRandomNumber()
{
  return ((double)rand() / ((double)(RAND_MAX)));
}

void read_matrix(int &n, mat &a, mat &b, string filename)
{
    ifstream file;
    file.open(filename);
    file >> n;
    int i, j;
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            file >> a[i][j];
    for(i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            file >> b[i][j];
    file.close();
}

void get_levels(vector<vector<Node*>> &levels, Node* node){
    if(levels.size()<=node->depth){
        levels.push_back(vector<Node*>());
    } 
    levels[node->depth].push_back(node);
    for(int i=0;i<node->children.size();i++){
        get_levels(levels, node->children[i]);
    }
}

double calc_agg_cost(double level_dis_mat[][N_MAX][N_MAX], double gamma, Node* tree){
    vector<vector<Node*>> levels;
    get_levels(levels, tree);
    double cost = 0;
    double times = 1;
    int num_levels = levels.size();
    // vector<double> avgs({0, 170827915.1, 36788362615.5, 1931046872871.8, 12153108876535.5, 20005951465490.1});
    // vector<double> stds({0, 304550.4, 1156040.0, 1893197.5, 3094937.7, 4303701.3});
    for(int l=num_levels-1;l>0;l--){
        for(int i=0;i<levels[l].size();i++) ids[i] = levels[l][i]->id;
        double level_cost = 0;
        int n = levels[l].size();
        // omp_set_num_threads(48);
        #pragma omp parallel for reduction (+:level_cost)
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                level_cost += level_bar_mat[l][i][j] * level_dis_mat[l][ids[i]][ids[j]];
            }
        }
        cost += times * ((level_cost - avgs[l]) / stds[l]);
        times *= gamma;
    }
    return cost;
}

vector<double> calc_level_bar(double level_dis_mat[][N_MAX][N_MAX], Node* tree){
    vector<vector<Node*>> levels;
    get_levels(levels, tree);
    double cost = 0;
    double times = 1;
    int num_levels = levels.size();

    vector<double> bars;
    
    // for(int i=0;i<levels[0].size();i++) cout << level_dis_mat[0][0][i] << " ";
    // cout << endl;
    // for(int i=0;i<levels[1].size();i++) cout << level_dis_mat[1][0][i] << " ";
    // cout << endl;
    // for(int i=0;i<levels[2].size();i++) cout << level_dis_mat[2][0][i] << " ";
    // cout << endl;

    for(int l=num_levels-1;l>0;l--){
        int n = levels[l].size();
        
        for(int i=0;i<levels[l].size();i++) ids[i] = levels[l][i]->id;
        double level_cost = 0;
        // omp_set_num_threads(48);
        #pragma omp parallel for reduction (+:level_cost)
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                level_cost += level_bar_mat[l][i][j] * level_dis_mat[l][ids[i]][ids[j]];
            }
        }
        // cout << l << " " << level_cost << endl;
        bars.push_back(level_cost);
    }
    return bars;
}

void random_shuffle_hierarchy(Node* tree){
    if(tree->children.empty()) return;
    random_shuffle(tree->children.begin(), tree->children.end());
    for(int i=0;i<tree->children.size();i++) random_shuffle_hierarchy(tree->children[i]);
}


vector<double> test_average_level(int n, mat &a, mat &b, Node* tree) {
    // cout << "----testing average level start----" << endl;
    // b是相似度矩阵
    // get levels for tree
    setDepth(tree, 0);
    vector<vector<Node*>> levels;
    get_levels(levels, tree);
    int num_levels = levels.size();
    // 不改变最后一层的id
    for(int i=0;i<num_levels-1;i++){
        vector<Node*> level = levels[i];
        for(int j=0;j<level.size();j++){
            level[j]->id = j;
        }
    }
    // build disance matrix for levels
    for(int l=num_levels-1; l>=0; l--){
        vector<Node*> level = levels[l];
        int dim = level.size();
        if(l==num_levels-1){
            for(int i=0;i<dim;i++){
                for(int j=0;j<dim;j++){
                    if(i==j) continue; // level_dis_mat 和 level_bar_mat 对角线均为0
                    level_dis_mat[l][i][j] = b[i][j]+b[j][i];
                    level_bar_mat[l][i][j] = dim - abs(i - j);
                }
            }
        }
        else{
            for(int i=0;i<dim;i++){
                for(int j=0;j<dim;j++){
                    if(i==j) continue;
                    vector<int> ni_leaves_id = level[i]->leavesId;
                    vector<int> nj_leaves_id = level[j]->leavesId;
                    double dis = 0;
                    for(auto it1=ni_leaves_id.begin();it1!=ni_leaves_id.end();it1++){
                        for(auto it2=nj_leaves_id.begin();it2!=nj_leaves_id.end();it2++){
                            dis += b[*it1][*it2] + b[*it2][*it1];
                        }
                    }
                    level_dis_mat[l][i][j] = dis;
                    level_bar_mat[l][i][j] = dim - abs(i - j);
                }
            }
        }
    }

    double max_conf = -1;
    for(int l=0;l<num_levels;l++){
        for(int i=0;i<levels[l].size();i++){
            for(int j=0;j<levels[l].size();j++){
                if(max_conf<level_dis_mat[l][i][j]) {
                    max_conf = level_dis_mat[l][i][j];
                } 
            }
        }
    }
    // 将相似度矩阵转为距离矩阵, 对角线置0
    max_conf += 1000;
    // cout << "maxconf+1000 " << max_conf << endl;
    for(int l=0;l<num_levels;l++){
        for(int i=0;i<levels[l].size();i++){
            for(int j=0;j<levels[l].size();j++){
                if(i==j) level_dis_mat[l][i][j] = 0;
                else level_dis_mat[l][i][j] = max_conf - level_dis_mat[l][i][j];
            }
        }
    }

    random_shuffle_hierarchy(tree);
    setLeaves(tree);

    vector<double> result = calc_level_bar(level_dis_mat,  tree);
    return result;
}

void annealing_hierarchy(int n, mat &a, mat &b, vector<int> &best_perm, double &best_cost, int num_iters, int num_init_iters, Node* tree, double gamma, bool preordered)
{
    // cout << "----annealing hierarchy start----" << endl;
    // b是相似度矩阵
    // get levels for tree
    Node* best_tree;
    best_tree = deepCopy(tree);
    setLeaves(best_tree);
    setDepth(best_tree, 0);
    setDepth(tree, 0);
    vector<vector<Node*>> levels;
    get_levels(levels, tree);
    int num_levels = levels.size();
    // 不改变最后一层的id
    for(int i=0;i<num_levels-1;i++){
        vector<Node*> level = levels[i];
        for(int j=0;j<level.size();j++){
            level[j]->id = j;
        }
    }
    // build disance matrix for levels
    for(int l=num_levels-1; l>=0; l--){
        vector<Node*> level = levels[l];
        int dim = level.size();
        if(l==num_levels-1){
            // 最底层直接读b，不使用hierarchy底层id做索引
            for(int i=0;i<dim;i++){
                for(int j=0;j<dim;j++){
                    if(i==j) continue;
                    level_dis_mat[l][i][j] = b[i][j]+b[j][i];
                    level_bar_mat[l][i][j] = dim - abs(i - j);
                }
            }
        }
        else{
            for(int i=0;i<dim;i++){
                for(int j=0;j<dim;j++){
                    if(i==j) continue;
                    vector<int> ni_leaves_id = level[i]->leavesId;
                    vector<int> nj_leaves_id = level[j]->leavesId;
                    double dis = 0;
                    for(auto it1=ni_leaves_id.begin();it1!=ni_leaves_id.end();it1++){
                        for(auto it2=nj_leaves_id.begin();it2!=nj_leaves_id.end();it2++){
                            dis += b[*it1][*it2] + b[*it2][*it1];
                        }
                    }
                    level_dis_mat[l][i][j] = dis;
                    level_bar_mat[l][i][j] = dim - abs(i - j);
                }
            }
        }
    }

    double max_conf = -1;
    int ii, jj;
    for(int l=0;l<num_levels;l++){
        for(int i=0;i<levels[l].size();i++){
            for(int j=0;j<levels[l].size();j++){
                if(max_conf<level_dis_mat[l][i][j]) {
                    max_conf = level_dis_mat[l][i][j];
                    ii = i;
                    jj = j;
                } 
            }
        }
    }
    // 将相似度矩阵转为距离矩阵
    max_conf += 1000;
    // cout << "maxconf+1000 " << max_conf << endl;
    for(int l=0;l<num_levels;l++){
        for(int i=0;i<levels[l].size();i++){
            for(int j=0;j<levels[l].size();j++){
                if(i==j) level_dis_mat[l][i][j] = 0;
                else level_dis_mat[l][i][j] = max_conf - level_dis_mat[l][i][j];
            }
        }
    }

    vector<Node*> interval_nodes;
    get_interval_nodes(tree, interval_nodes);
    vector<Node*> multi_interval_nodes;
    for(int i=0;i<interval_nodes.size();i++){
        if(interval_nodes[i]->children.size()>1)
            multi_interval_nodes.push_back(interval_nodes[i]);
    }
    // 需确保输入中至少有一个多叉树
    interval_nodes.clear();
    interval_nodes.assign(multi_interval_nodes.begin(), multi_interval_nodes.end());
    // cout << "size of interval node " << interval_nodes.size()<< endl;
    // cout << interval_nodes[0]->children.size() << endl;

    int i, r, s;
    int interval_idx = 0;
    Node* interval, *tmp, *leaf_r, *leaf_s;

    double delta;
    int max_fail = 9942; // n * (n - 1) / 2;
    int num_fail, cur_iter;
    double dmin = __DBL_MAX__, dmax = 0;
    double t0, tf, beta, temp_found, cur_temp;

    double cost = calc_agg_cost(level_dis_mat, gamma, tree);
    double old_cost, new_cost;
    best_cost = cost;

    // cout << "initial cost: " << cost << endl;

    // cout << "before: level bar test" << endl;
    // calc_level_bar(level_dis_mat, tree);

    // cout << "initial perm" << endl;
    // for(int i=0;i<tree->leavesId.size();i++) cout << tree->leavesId[i] << " ";
    // cout << endl;

    for (cur_iter = 0; cur_iter < num_init_iters; cur_iter++)
    {
        // cout << cur_iter << endl;
        // just accept the random swap
        interval_idx = generateRandomNumber(0, interval_nodes.size()-1);
        interval = interval_nodes[interval_idx];
        if(interval->children.size()==1) {
            cur_iter --; // cout << "--" << endl;
            continue;
        }
        r = generateRandomNumber(0, interval->children.size() - 1);
        s = generateRandomNumber(0, interval->children.size() - 1);
        while (s == r) s = generateRandomNumber(0, interval->children.size() - 1);
        tmp = interval->children[r];
        interval->children[r] = interval->children[s];
        interval->children[s] = tmp;
        setLeaves(tree);

        // swap subtrees
        old_cost = cost;    
        new_cost = calc_agg_cost(level_dis_mat, gamma, tree);
        delta = new_cost - old_cost;

        if (delta > 0)
        {
            dmin = min(dmin, delta);
            dmax = max(dmax, delta);
        };

        if(preordered){
            tmp = interval->children[r];
            interval->children[r] = interval->children[s];
            interval->children[s] = tmp;
            setLeaves(tree);
        }

        cost = cost + delta;
    }

    // cout << "after init: level bar test" << endl;
    // calc_level_bar(level_dis_mat, tree);


    // if(dmin==__DBL_MAX__) cout << "DBL_MAX" << endl;
    t0 = dmin + (dmax - dmin) / 10.0;
    tf = dmin;
    // cout << "tf " << tf << endl;
    beta = (t0 - tf) / (num_iters * t0 * tf);

    int epochs = num_iters - num_init_iters;
    if(preordered) {
        cout << "preordered" << endl;
        epochs = 10000;
        beta = (t0 - tf) / (epochs * t0 * tf);
        beta *= 100;
    }

    num_fail = 0;
    temp_found = t0;
    cur_temp = t0;
    // change enumerate to random select
    // r = 0;
    // s = 1;
    // cout << "dmin " << dmin << endl;
    // cout << "dmax " << dmax << endl;
    // cout << "beta " << beta << endl;
    // cout << "t0 " << t0 << endl;
    // cout << "tf " << tf << endl;
    // cout << "init iters done" << endl;
    vector<double> threadholds;
    int good_acc = 0;
    int bad_acc = 0;
    int fail_acc = 0;
    int refuse = 0;
    int no_incre = 0;
    int max_no_incre = 100000;
    int best_cost_update_num = 0;

    // cout << "n " << interval_nodes[0]->children.size() << endl;
    cur_iter = 0;
    for (cur_iter = 0; cur_iter < epochs; cur_iter++)
    {
        if((cur_iter+1)%1000==0) {
            // cout << cur_iter << endl;
            // cout << "cur t " << cur_temp << endl;
            // cout << "good " << good_acc << " " << good_acc/double(1000) << endl;
            // cout << "bad " << bad_acc << " " << bad_acc/double(1000) << endl;
            // cout << "refuse " << refuse << " " << refuse/double(1000) << endl;
            // cout << "fail " << fail_acc << " " << fail_acc/double(1000) << endl;
            // cout << "best_cost_update_num " << best_cost_update_num << endl;
            good_acc=0;
            bad_acc=0;
            refuse=0;
            fail_acc=0;
            best_cost_update_num = 0;
        }
        // if(preordered) cout << cur_iter << endl;
        cur_temp = cur_temp / (1.0 + beta * cur_temp);

        interval_idx = generateRandomNumber(0, interval_nodes.size()-1);
        interval = interval_nodes[interval_idx];
        if(interval->children.size()==1) {
            cur_iter--;  // cout << "--" << endl;
            continue;
        }
        r = generateRandomNumber(0, interval->children.size() - 1);
        s = generateRandomNumber(0, interval->children.size() - 1);
        while (s == r) s = generateRandomNumber(0, interval->children.size() - 1);

        // cout <<"r s" << endl;
        // cout << r << " " << s << endl;
        // cout << interval->children[r]->id << " " << interval->children[s]->id << endl;
        // cout << "initial perm" << endl;
        // for(int i=0;i<n;i++) cout << perm[i] << " ";
        // cout << endl;
        
        tmp = interval->children[r];
        interval->children[r] = interval->children[s];
        interval->children[s] = tmp;
        setLeaves(tree);
        
        // swap subtrees
        old_cost = cost;    
        new_cost = calc_agg_cost(level_dis_mat, gamma, tree);
        delta = new_cost - old_cost;

        threadholds.push_back(exp(-double(delta) / cur_temp));
        // cout << exp(-double(delta) / cur_temp) << endl;
        double random_number = generateDoubleRandomNumber();
        if ((delta < 0) ||  (random_number < exp(-double(delta) / cur_temp)) || max_fail == num_fail)
        {
            // if(delta>0){
            //     cout << ">0 accept prop begin " << exp(-double(delta) / cur_temp) << endl;
            //     cout << "delta " << delta << endl;
            //     cout << "curtemp " << cur_temp << endl;
            //     cout << "delta / curtemp " << double(delta) / cur_temp << endl;
            //     cout << ">0 accept prop end " << endl;
            // }

            if(delta<0) good_acc++;
            if(delta>=0 && (random_number < exp(-double(delta) / cur_temp))) bad_acc++;
            if( !((delta < 0) ||  (random_number < exp(-double(delta) / cur_temp))) && (max_fail == num_fail)) fail_acc++;

            // if(delta>0) cout << "worse but accept " << exp(-double(delta) / cur_temp) << endl;
            // accept, already changed the hierarchy
            cost = cost + delta;
            // cout << "num_fail update " << num_fail << " -> 0" << endl;
            num_fail = 0;
            // cout << "accept" << endl;
            // for(int i=0;i<n;i++) cout << perm[i] << " ";
            // cout << endl;
        }
        else {
            refuse++;
            // refuse, recover the hierarchy
            num_fail = num_fail + 1;
            // cout << "num_fail " << num_fail << endl;
            tmp = interval->children[r];
            interval->children[r] = interval->children[s];
            interval->children[s] = tmp;
            setLeaves(tree);
            // cout << "refuse" << endl;
            // for(int i=0;i<n;i++) cout << perm[i] << " ";
            // cout << endl;
        }  
        if (max_fail == num_fail)
        {
            // cout << "max_fail = num_fail" << endl;
            beta = 0;
            cur_temp = temp_found;
        }
        if (cost < best_cost)
        {
            // cout << "Iteration = " << cur_iter << "old Cost = " << best_cost << " new cost " << new_cost << " delta cost " << best_cost - cost << endl;
            best_cost = cost;
            for (int i = 0; i < n; i++)
                best_perm[i] = tree->leavesId[i];
            // for(int i=0;i<n;i++){
            //     cout << best_perm[i] << " ";
            // }
            // cout << endl;
            best_tree = nullptr;
            best_tree = deepCopy(tree);
            setLeaves(best_tree);
            setDepth(best_tree, 0);
            temp_found = cur_temp;
            
            no_incre = 0;
            best_cost_update_num += 1;
        } else{
            no_incre ++;
        }
        // if(no_incre==max_no_incre){
        //     cout << "no incre in " << max_no_incre << "epochs, cur epoch " << cur_iter << endl;
        //     break;
        // }
        // if(cur_iter==epochs){
        //     cout << "iter achieve max epoch " << cur_iter << endl;
        //     break;
        // }
    }
    // cout << "temp_found " << temp_found << endl;
    // cout << "cur temp " << cur_temp << endl;

    //   cout << "Best solution found : \n";
    //   for (i = 0; i < n; i++) cout << best_perm[i] << ' ';
    //   cout << endl;;

    // cout << "after: level bar test" << endl;
    // calc_level_bar(level_dis_mat, tree);
    // cout << "best cost " << best_cost << endl;
    // cout << "----annealing hierarchy done----" << endl;
}

Node* hierarchyToTree(map<string, vector<string>>& hierarchy) {
    map<string, Node*> nodes;
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
    
    // note: name of the root of the hierarchy is "root"
    return nodes["root"];
}

void setId(Node* tree, map<string, int>& name2id) {
    int inter_start_id = INTERNAL_NODE_ID_START;
    int start_id = 0; // used when name2id is empty
    deque<Node*> que;
    que.push_back(tree);
    while(que.size()>0){
        Node* p = que[0];
        que.pop_front();
        if(p->isLeaf()){
            if(name2id.count(p->name)!=0)
                p->id = name2id[p->name];
            else {
                // name2id is empty, leaves are assigned id [0,n-1] from left to right
                p->id = start_id;
                start_id++;
            }
        } else {
            // id of interval nodes is of no use
            p->id = inter_start_id;
            inter_start_id++;
            que.insert(que.end(), p->children.begin(), p->children.end());
        }
    }
}

void print_tree(Node* node){
    cout << node->name << " " << node->id << endl;
    for(int i=0;i<node->children.size();i++){
        cout << node->children[i]->name << " ";
    }
    cout << endl;
    for(int i=0;i<node->children.size();i++){
        print_tree(node->children[i]);
    }
}

void read_hierarchy(map<string, vector<string>> &hierarchy, string hierarchy_path){ 
    ifstream file; 
    file.open(hierarchy_path);
    string tmp_1, tmp_2;
    string line;
    vector<string> children;
    while(getline(file, line)){
        istringstream iss(line);
        iss >> tmp_1;
        children.clear();
        while(iss>>tmp_2){
            children.push_back(tmp_2);
        }
        hierarchy[tmp_1] = children;
    }
}

void read_name2id(map<string, int>& name2id, string name2id_path) {
    ifstream file;
    file.open(name2id_path);
    string tmp_name;
    int tmp_id;
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        iss >> tmp_name;
        iss >> tmp_id;
        // cout << tmp_id << endl;
        name2id[tmp_name] = tmp_id;
    }
    file.close();

    // ifstream file; 
    // file.open(hierarchy_path);
    // string tmp_1, tmp_2;
    // string line;
    // vector<string> children;
    // while(getline(file, line)){
    //     istringstream iss(line);
    //     iss >> tmp_1;
    //     children.clear();
    //     while(iss>>tmp_2){
    //         children.push_back(tmp_2);
    //     }
    //     hierarchy[tmp_1] = children;
    // }
}


vector<int> getQAPReordering_file(double gamma, bool has_hierarchy, string mat_path, string hierarchy_path, string name2id_path, int restart, int num_iters, bool preordered, vector<double> data_avgs, vector<double> data_stds){
    cout << fixed << setprecision(12);
    srand(101);

    avgs.assign(data_avgs.begin(), data_avgs.end());
    stds.assign(data_stds.begin(), data_stds.end());

    Node* tree = nullptr;
    if(has_hierarchy){
        read_matrix(n, a, b, mat_path);
        read_hierarchy(hierarchy, hierarchy_path);
        read_name2id(name2id, name2id_path);
    } else {
        cout << "no hierarchy" << endl;
        read_matrix(n, a, b, mat_path);
        vector<string> strs;
        for(int i=0;i<n;i++) strs.push_back(to_string(i));
        hierarchy["root"] = strs;
    }
    tree = hierarchyToTree(hierarchy);
    setId(tree, name2id); 
    setLeaves(tree); 

    // int num_iters = 35000;
    int num_init_iters = num_iters/100; // Connolly proposes nb_iterations/100
    double tmp_cost;
    vector<int> tmp_perm;
    vector<int> best_perm;
    double best_cost = 2e20;

    for (int i = 0; i < restart; i++)
    {
        auto start = std::chrono::steady_clock::now();
        Node* tmp_tree = deepCopy(tree);
        if(!preordered) 
            random_shuffle_hierarchy(tmp_tree);
        setLeaves(tmp_tree);
        tmp_perm.assign(tmp_tree->leavesId.begin(), tmp_tree->leavesId.end());
        annealing_hierarchy(n, a, b, tmp_perm, tmp_cost, num_iters, num_init_iters, tmp_tree, gamma, preordered);
        if (best_cost > tmp_cost){
            best_cost = tmp_cost;
            best_perm.assign(tmp_perm.begin(), tmp_perm.end());
        }
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double runtime = double(diff.count()) * 0.000001;
        cout << "time " << runtime << endl;
        cout << endl;

    }       
    cout << "best cost in all " << best_cost << endl;
    // 只能返回best_perm，因为hierarchy对应矩阵B
    // vector<int> best_perm(best_perm, best_perm+n);
    // return best_perm;
    // cout << "best perm in all" << endl;
    // for(int i=0;i<best_perm.size();i++) cout << best_perm[i] << " ";
    // cout << endl;
    return best_perm;
}

vector<int> getQAPReordering(double gamma, bool has_hierarchy, int N, vector<vector<double>> B, string hierarchy_path, string name2id_path, int restart, int num_iters, bool preordered, vector<double> data_avgs, vector<double> data_stds){
    cout << fixed << setprecision(12);
    srand(101);

    avgs.assign(data_avgs.begin(), data_avgs.end());
    stds.assign(data_stds.begin(), data_stds.end());

    n = N;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            b[i][j] = B[i][j];
        }
    }
    Node* tree = nullptr;
    if(has_hierarchy){
        // read_matrix(n, a, b, mat_path);
        read_hierarchy(hierarchy, hierarchy_path);
        read_name2id(name2id, name2id_path);
    } else {
        // cout << "no hierarchy" << endl;
        // read_matrix(n, a, b, mat_path);
        vector<string> strs;
        for(int i=0;i<n;i++) strs.push_back(to_string(i));
        hierarchy["root"] = strs;
    }
    tree = hierarchyToTree(hierarchy);
    setId(tree, name2id); 
    setLeaves(tree); 

    // int num_iters = 35000;
    int num_init_iters = num_iters/100; // Connolly proposes nb_iterations/100
    double tmp_cost;
    vector<int> tmp_perm;
    vector<int> best_perm;
    double best_cost = 2e20;

    for (int i = 0; i < restart; i++)
    {
        auto start = std::chrono::steady_clock::now();
        Node* tmp_tree = deepCopy(tree);
        if(!preordered) 
            random_shuffle_hierarchy(tmp_tree);
        setLeaves(tmp_tree);
        tmp_perm.assign(tmp_tree->leavesId.begin(), tmp_tree->leavesId.end());
        annealing_hierarchy(n, a, b, tmp_perm, tmp_cost, num_iters, num_init_iters, tmp_tree, gamma, preordered);
        if (best_cost > tmp_cost){
            best_cost = tmp_cost;
            best_perm.assign(tmp_perm.begin(), tmp_perm.end());
        }
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double runtime = double(diff.count()) * 0.000001;
        // cout << "time " << runtime << endl;
        // cout << endl;

    }       
    // cout << "best cost in all " << best_cost << endl;
    // 只能返回best_perm，因为hierarchy对应矩阵B
    // vector<int> best_perm(best_perm, best_perm+n);
    // return best_perm;
    // cout << "best perm in all" << endl;
    // for(int i=0;i<best_perm.size();i++) cout << best_perm[i] << " ";
    // cout << endl;
    return best_perm;
}

vector<vector<double>> getAvgAndStd(bool has_hierarchy, string mat_path, string hierarchy_path, string name2id_path, int test_tree_num){
    cout << fixed << setprecision(1);
    srand(101);

    Node* tree = nullptr;
    if(has_hierarchy){
        read_matrix(n, a, b, mat_path);
        read_hierarchy(hierarchy, hierarchy_path);
        read_name2id(name2id, name2id_path);
    } else {
        cout << "no hierarchy" << endl;
        read_matrix(n, a, b, mat_path);
        vector<string> strs;
        for(int i=0;i<n;i++) strs.push_back(to_string(i));
        hierarchy["root"] = strs;
    }
    tree = hierarchyToTree(hierarchy);
    setId(tree, name2id); 
    setLeaves(tree); 

    vector<vector<double>> all_results; // num_tree * num_level

    for (int i = 0; i < test_tree_num; i++)
    {
        // cout << i << endl;
        Node* tmp_tree = deepCopy(tree);
        random_shuffle_hierarchy(tmp_tree);
        setLeaves(tmp_tree);
        all_results.push_back(test_average_level(n, a, b, tmp_tree));
    }       

    int num_levels = all_results[0].size();

    vector<vector<double>> level_results; // num_level * num_tree
    for(int i=0;i<num_levels;i++){
        level_results.push_back(vector<double>());
        for(int j=0;j<test_tree_num;j++){
            level_results[i].push_back(all_results[j][i]);
        }
    }

    vector<double> res_avgs;
    vector<double> res_stds;
    for(int i=0;i<num_levels;i++){
        res_avgs.push_back(getAvg(level_results[i]));
        res_stds.push_back(getStd(level_results[i]));
    }

    vector<vector<double>> result;
    result.push_back(res_avgs);
    result.push_back(res_stds);
    return result;
}


int main() {
    // string path = "/data/fengyuan/ConfusionMatrix/backend/data/reordering/pkgC/data_test/";
    // srand(unsigned(std::time(0)));
    srand(101);
    
    string path = "/data/fengyuan/72/ConfusionMatrix/backend/data/reordering/data/coco/100/";
    // calc average bar
    vector<vector<double>> result = getAvgAndStd(true, path+"matrix.txt", path+"hierarchy.txt", path+"name2id.txt", 5000);
    cout << "avg" << endl;
    for(int i=0;i<result[0].size();i++){
        cout << result[0][i] << endl;
    }
    cout << "std" << endl;
    for(int i=0;i<result[1].size();i++){
        cout << result[1][i] << endl;
    }

    // annealing
    // int num_test_cnt=1;
    // vector<double> runtimes;
    // vector<double> results;
    // double runtime;
    // for (int i = 0; i < num_test_cnt; i++)
    // {
    //     auto start = std::chrono::steady_clock::now();
    //     getQAPReordering_file(0.5, true, path+"case_1000.txt", path+"hierarchy_1000_pre.txt", path+"name2id_1000.txt", 1, 35000, true);
    //     // getQAPReordering_file(0.5, true, path+"case_1000.txt", path+"hierarchy_1000.txt", path+"name2id_1000.txt", 1, 35000, false);
    //     // getQAPReordering_file(0, false, path+"casetmp.txt", path+"hierarchy_261.txt", path+"name2id_261.txt", 1);
    //     auto end = std::chrono::steady_clock::now();
    //     auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //     double runtime = double(diff.count()) * 0.000001;
    //     runtimes.push_back(runtime);
    //     // results.push_back(cur_best);
    // }
    // cout << "cost avg " << getAvg(results) << endl;
    // cout << "cost var " << getVar(results) << endl;
    // cout << "time avg " << getAvg(runtimes) << endl;
    return 0;
}

PYBIND11_MODULE(agg_sa, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    // m.def("add", &add, "a function that adds two numbers");
    m.def("getQAPReordering_file", &getQAPReordering_file, "");
    m.def("getQAPReordering", &getQAPReordering, "");
    m.def("getAvgAndStd", &getAvgAndStd, "");
}