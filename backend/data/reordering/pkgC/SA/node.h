#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
using namespace std;


class Node {
    public:
        int id;
        string name;
        vector<Node*> children;
        double score;
        int depth = -1;

        vector<Node*> leaves;
        vector<int> leavesId;

        Node(){};
        Node(const string& name):name(name){id=-1;}
        Node(int id, const string& name):id(id), name(name){};
        Node(const Node& n);
        bool isLeaf();
        vector<int> getLeavesId();
};

bool Node::isLeaf() {
    return children.size()==0;
}

Node::Node(const Node& n){
    id = n.id;
    name = n.name;
    children = n.children;
}

vector<int> Node::getLeavesId(){
    vector<int> ids;
    for(auto l: leaves){
        ids.push_back(l->id);
    }
    this->leavesId = ids;
    return ids;
}

Node* deepCopy(const Node* ori_tree) {
    Node* new_tree = new Node();
    new_tree->id = ori_tree->id;
    new_tree->name = ori_tree->name;
    for(auto ori_c: ori_tree->children){
        Node* new_c = deepCopy(ori_c);
        new_tree->children.push_back(new_c);
    }
    return new_tree;
}

void clearLeaves(Node* tree){
    if(!tree) return;
    tree->leaves.clear();
    for(auto c: tree->children) clearLeaves(c);
}

void setLeaves(Node* tree){
    tree->leaves.clear();
    tree->leavesId.clear();
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

void setDepth(Node* tree, int depth){
    if(tree==nullptr) return;
    tree->depth = depth;
    for(int i=0;i<tree->children.size();i++){
        setDepth(tree->children[i], depth+1);
    }
}