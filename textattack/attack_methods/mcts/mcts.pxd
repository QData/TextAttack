# Cython Imports
from libc.stdint cimport uint32_t
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool

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

cdef class MCSearchTree:
    cdef:
        Node root
        Node* current_node
        uint32_t max_depth
        object transformation_call
        object model_call

    cpdef void run_search(self)
    cdef Node selection(self)
    cdef void expansion(self)
    cdef void simulation(self)
    cdef void backprop(self)
    cdef double UCB(self, Node& node, double nvisits_parent)
    cdef double UCB_tuned(self)