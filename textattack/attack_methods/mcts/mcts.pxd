# Cython Imports
from libc.stdint cimport uint32_t
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "node.hpp":
    cdef cppclass Node:
        set[uint32_t] state
        uint32_t depth
        uint32_t num_visits
        double value
        Node* parent
        map[uint32_t, Node*] children

        Node()
        Node(uint32_t depth, Node* parent)
        bool isLeaf()

cdef class TransformationCache:
    """
    Used for storeing queried transformation for later repeated uses.

    Members:
        cache: Dictionary where key is unsigned int representing the index of the word we want to transform
                and value is list of TokenizedText objects returned by `get_transformations`

    """
    cdef:
        dict cache


cdef class MCSearchTree:
    """
    C-extension for actually carrying out Monte Carlo Tree Search.

    Members:
        max_depth: Maximum depth of the search tree
        orignal_text (TokenizedText): orignal text that is target of attack
        transformation_call (Python callable): get_transformation method
        model_call (Python callable): call_model method

    """

    cdef:
        Node root
        uint32_t max_depth
        object orignal_text
        object transformation_call
        object model_call

    cpdef void run_search(self, uint32_t max_iter)
    cdef uint32_t choose_random_action(self)
    cdef void reward(self)
    cdef Node* selection(self)
    cdef void expansion(self)
    cdef void simulation(self)
    cdef void backprop(self)
    cdef double UCB(self, Node& node, double nvisits_parent)
    cdef double UCB_tuned(self)