# Cython Imports
from cython.operator cimport dereference as deref
from cython.operator cimport postincrement as postinc
from libc.math cimport HUGE_VAL, log, sqrt

cdef class MCSearchTree:
    def __cinit__(self, uint32_t max_depth, object transformation_call, object model_call):
        self.root = Node(0, NULL)
        self.current_node = &self.root
        self.max_depth = max_depth
        self.transformation_call = transformation_call
        self.model_call = model_call

    cdef void reward(self):
        pass

    cdef void backprop(self):
        pass

    cdef void simulation(self):
        pass

    cdef void expansion(self):

        

    cdef double UCB(self, Node& node, double nvisits_parent):
        return node.value + sqrt(self.ucb_C * log(nvisits_parent) / node.num_visits)

    cdef double UCB_tuned(self):
        # to be implemented
        pass

    cdef Node selection(self):
        self.current_node = &self.root

        cdef Node* best_child = NULL
        cdef double best_ucb_value = -1 * HUGE_VAL
        cdef double ucb_value
        cdef map[uint32_t, Node*].iterator it

        while not current_node.isLeaf() or current_node.depth <= self.tree.max_depth:

            # Iterate through the children to find the best child to visit
            it = current_node.children.begin()
            while(it != current_node.children.end()):
                ucb_value = self.UCB(deref(deref(it).second), current_node.num_visits)

                if ucb_value > best_ucb_value:
                    best_child = deref(it).second
                    best_ucb_value = ucb_value

                postinc(it)

            current_node = deref(best_child)

        return current_node

    cpdef void run_search(self, uint32_t max_iter):
        cdef Node last_node

        for i in range(max_iter):
            last_node = self.selection()
            self.expansion()
            self.simulation()
            self.backprop()

