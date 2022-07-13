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
        vector<Node*> leaves;
        vector<int> leavesId;

        int tmp_id=-1;

        Node(){};
        Node(const string& name):name(name){id=-1;}
        Node(int id, const string& name):id(id), name(name){};
        Node(const Node& n);
        bool isLeaf();
        vector<int> getSortedLeavesId();
};

bool Node::isLeaf() {
    return children.size()==0;
}

Node::Node(const Node& n){
    id = n.id;
    name = n.name;
    children = n.children;
    leaves = n.leaves;
}

vector<int> Node::getSortedLeavesId(){
    vector<int> ids;
    for(auto l: leaves){
        ids.push_back(l->id);
    }
    sort(ids.begin(), ids.end());
    return ids;
}

