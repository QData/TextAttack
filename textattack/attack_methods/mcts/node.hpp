#include <set>
#include <map>

typedef unsigned int uint;

class Node {
    /**
     * Represents a node in MC search tree.
     * Members:
     *   state: Set of words selected to be transformed for attack. Represented as set of indices.
     *   depth: Depth of the node in search tree. Root has depth 0
     *   num_visists: Number of times we have visited this node during MCTS
     *   value: Value of the state.
     *   parent: Parent node
     *   children: Children of the node
     */

public:
    std::set<uint> state; 
    uint depth;
    uint num_visits;
    double value;
    Node* parent;
    std::map<uint, Node*> children;

    Node(): state(std::set<uint>()), depth(0), num_visits(0), 
            value(0.0), parent(NULL), children(std::map<uint, Node*>()) {}
    Node(uint depth, Node* parent): 
            state(std::set<uint>()), depth(depth), num_visits(0), 
            value(0.0), parent(parent), children(std::map<uint, Node*>()) {}
    bool isLeaf();
};

bool Node::isLeaf() {
    if (children.size() == 0)
        return true;
    else
        return false;
}