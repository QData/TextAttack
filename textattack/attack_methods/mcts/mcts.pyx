# Cython Imports
from cython.operator cimport dereference as deref
from cython.operator cimport postincrement as postinc
from libc.math cimport HUGE_VAL, log, sqrt
from libc.stdlib cimport rand, RAND_MAX

cdef class TransformationCache:
    def __cinit__(self):
        self.cache = {}

cdef class MCSearchTree:
    def __cinit__(self, 
        object original_text, 
        uint32_t max_depth
        object transformation_call, 
        object model_call

    ):
        self.root = Node(0, NULL)
        self.max_depth = min(max_depth, len(original_text.words))
        self.transformation_call = transformation_call
        self.model_call = model_call
        self.original_text = original_text
        self.current_text = original_text


        self.pseudo_state = {}

    cdef uint32_t choose_random_action(self):
        RAND_MAX = len(self.original_text.words)
        return rand()

    cdef object play_action(uint32_t action):
        """
        Randomly generate a transformation for the chosen word of current text

        Args:
            action: index of the word to transform

        Returns: TokenizedText object
        """

        list transformations = self.transformations_call(current_text, indices_to_replace=[action])
        RAND_MAX = len(transformations)
        return transformations[rand()]

    cdef void reward(self):
        pass

    cdef void backprop(self):
        pass

    cdef void simulation(self):
        pass

    cdef void expansion(self):
        cdef uint32_t random_action = self.choose_random_action()
        cdef object transformed_text = 
        pass

    cdef double UCB(self, Node& node, double nvisits_parent):
        return node.value + sqrt(self.ucb_C * log(nvisits_parent) / node.num_visits)

    cdef double UCB_tuned(self):
        # to be implemented
        pass

    cdef Node* selection(self):
        cdef Node* current_node = &self.root

        cdef Node* best_child = NULL
        cdef double best_ucb_value = -1 * HUGE_VAL
        cdef double ucb_value
        cdef map[uint32_t, Node*].iterator it

        while not deref(current_node).isLeaf() or deref(current_node).depth <= self.tree.max_depth:

            # Iterate through the children to find the best child to visit
            it = deref(current_node).children.begin()
            while(it != deref(current_node).children.end()):
                ucb_value = self.UCB(deref(deref(it).second), deref(current_node).num_visits)

                if ucb_value > best_ucb_value:
                    best_child = deref(it).second
                    best_ucb_value = ucb_value

                postinc(it)

            current_node = best_child

        return current_node

    cpdef void run_search(self, uint32_t max_iter):
        cdef Node* current_node

        for i in range(max_iter):

            current_node = self.selection()
            self.expansion()
            self.simulation()
            self.backprop()

