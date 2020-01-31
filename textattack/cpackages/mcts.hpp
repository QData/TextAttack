#include <vector>


/**
 * Represents state in our MCTS tree search
 * Uses boolean vector to represent whether the word at index is selected for attack transformation
 * 0 - Not selected
 * 1 - Selected for attack
 */
class State {
    std::vector<bool> state;

public:
    State();
    State(int n): state(std::vector<bool>(n)) {}

    std::vector<bool> getState() { return state; };
    void setState(bool b, size_t i) { state[i] = b; };
}

class Node {
    State current_state;

    public:
        Node()
        Node(bool [] cs) curr_selections(cs);
};

class SearchTree {
    Node root;
  public:
    Tree() root(Node())
    Tree()
};


class MCTS {
    std::string reward_type;
    int max_iter;
    int max_words_changd;
    double all_time_best = -1e9
    std::vector<size_t> best_feature;

public:
    MCTS()
  



};

//TODO extend MCTS-RAVE