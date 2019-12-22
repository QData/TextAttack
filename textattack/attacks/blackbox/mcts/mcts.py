from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.blackbox import BlackBoxAttack
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

class dotdict(dict):
    """ dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Node:
    """ Helper node class used in implementation of MCTS"""
    def __init__(self,feature_set):
        self.feature_set = feature_set
        self.f_size = self.feature_set.sum()
        self.childrens = {}
        self.T_f = .0
        self.av = .0
        self.allowed_features = feature_set.nonzero()
        self.lrave_count = np.array(feature_set.shape)
        self.lrave_reward = np.array(feature_set.shape).astype(float)
        self.lrave_variance = np.array(feature_set.shape).astype(float)
        self.lrave_score = np.array(feature_set.shape).astype(float)

class Tree:
    """ Helper tree class used in implementation of MCTS; dictionary of nodes"""
    def __init__(self,nsize):
        self.tree = {}
        root = np.array([False] * nsize,dtype=bool)
        self.tree[root.tobytes()] = Node(root)

    def find(self,feature_set):
        """ Returns node containing features_set if present in tree"""
        if feature_set.tobytes() in self.tree:
            return self.tree[feature_set.tobytes()]
        else:
            return None

    def save(self,feature_set,node):
        """ Adds node with feature_set to tree"""
        self.tree[feature_set.tobytes()] = node

class MCTS(BlackBoxAttack):
    """ Uses Monte Carlo Tree Search (MCTS) to attempt to find the most important words in an input,
    params: targeted (bool, unimplemented!), valuefunction (str), power (int), nplayout (int)
    """
    def __init__(self, model, transformations=[], rewardvalue='combined', numiter=4000, max_words_changed=10):
        super().__init__(model)
        self.transformation = transformations[0]
        self.params = dotdict({})
        self.params.get_reward = None
        self.params.b = 0.26 #b < 1
        self.params.cl = 0.5
        self.params.ce = 0.5
        self.params.max_depth = 0
        self.params.nrand = 50
        self.gRAVE = None
        self.tree = None
        self.alltimebest = 0
        self.bestfeature = []
        self.rewardvalue = rewardvalue
        self.numiter = numiter
        self.max_words_changed = max_words_changed

    def update_Node(self, node, fi, reward_V):
        """ Updates the node's scores based on how its feature set performed against the model"""
        node.av = (node.av * node.T_f + reward_V)/(node.T_f+1)
        node.T_f += 1
        lrave_score_pre = node.lrave_score[fi]
        node.lrave_score[fi] = (node.lrave_score[fi] * node.lrave_count[fi] + reward_V) / (node.lrave_count[fi] + 1)
        node.lrave_variance[fi] = math.sqrt( ((reward_V - node.lrave_score[fi]) * (reward_V - lrave_score_pre) + node.lrave_count[fi] * node.lrave_variance[fi] * node.lrave_variance[fi])/(node.lrave_count[fi]+1))
        node.lrave_count[fi] = node.lrave_count[fi] + 1

    def update_gRAVE(self, F, reward_V):
        """ Update gRAVE score for each feature of feature subset F, by adding the reward_V the the score 
        gRAVE[0] -> Count 
        gRAVE[1] -> Score
        """
        for fi in range(self.gRAVE[0].shape[0]):
            if F[fi]:
                self.gRAVE[0][fi] = (self.gRAVE[0][fi]*self.gRAVE[1][fi] + reward_V)/(self.gRAVE[1][fi] + 1)
                self.gRAVE[1][fi] += 1 


    def UCB(self, node):
        """ Upper confidence bound calculation to help balance exploration/exploitation tradeoff"""
        d = len(node.allowed_features)  # feature subset size
        f = node.feature_set.shape[0] # number of features
        nrand = self.params.nrand

        if (node.T_f < nrand): # we perform random exploration fort the first 50 visits
            return -1 # -1 indicate that we don't to want choose any feature

        if (pow((node.T_f+1), self.params.b)-pow(node.T_f, self.params.b)>1):
            if not node.allowed_features:
                return -1
            else:
                for ft in range(f):
                    if not ft in allowed_features:
                        beta = self.params.cl/(self.params.cl + node.lrave_count[ft])
                        rst = (1-beta) * self.lrave_score[ft] + beta * self.gRAVE[0][ft]
                        if rst>nowbest:
                            ft_now = ft
                            nowbest = rst
                node.allowed_features.append(ft)
        else:
            return -1        

        UCB_max_score = 0
        UCB_max_feature = 0
        for next_node in node.allowed_features: # computing UCB for each feature
            UCB_Score = node.mu_f[next_node] + math.sqrt( params.ce*log(node.T_F)/node.t_f[next_node]  *  min(0.25 ,  pow(node.sg_f[next_node],2) + math.sqrt(2*math.log(node.T_F)/node.t_f[fi]) ))
            if UCB_Score>UCB_max_feature:
                UCB_max_feature = UCB_Score
                UCB_max_feature = next_node
        return UCB_max_feature


    def update_Tree_And_Get_Address(self, current_features):
        """ Updates MCTS tree to contain the set of current features if necessary, returning the newly created or found node"""
        if not self.tree.find(current_features):
            node = Node(current_features)
            self.tree.save(current_features, node)
        else:
            node = self.tree.find(current_features)
        return node

    def iterate(self, current_features, depth):
        """ Perform one iteration of MCTS search"""
        if depth>=self.params.max_depth:
            reward_V = self.params.get_reward(current_features)
            self.update_gRAVE(current_features, reward_V)
        else:
            next_node = self.update_Tree_And_Get_Address(current_features)

            if (next_node.T_f != 0):
                fi = self.UCB(next_node)
                if (fi == -1): # it means that no feature has been selected and that we are going to perform random exploration
                    depth = current_features.sum()
                    reward_V = self.iterate_random(self.tree, current_features)
                    self.update_gRAVE(current_features, reward_V)
                else: #add the feature to the feature set
                    current_features[fi] = True
                    reward_V = self.iterate(current_features, depth+1)
            else:
                depth_now = current_features.sum()
                reward_V = self.iterate_random(self.tree, current_features)
                self.update_gRAVE(current_features, reward_V)
                fi = -1 # indicate that random exploration has been performed and thus no feature selected
            self.update_Node(next_node, fi, reward_V)
        return reward_V

    def iterate_random(self, tree, current_features):
        """ Choose a random feature that is not already in the feature subset, and put its value to one (and not the stopping feature)"""
        f_num = current_features.shape[0]
        f_size  = current_features.sum()
        while (f_size < self.params.max_depth):
            if (f_num<=f_size):
                break
            t = 0
            it =  int(np.random.rand() * (f_num-f_size))
            for i in range(f_num):
                if not current_features[i] and t==it:
                    it = i
                    break
                elif not current_features[i]:
                    t = t + 1
            current_features[it] = True
            f_size += 1

        return self.params.get_reward(current_features)

    def runmcts(self, rewardfunc, maxdepth, lcount, nsize):
        """ Runs lcount iterations of MCTS"""
        self.params.get_reward = rewardfunc
        self.params.max_depth = maxdepth
        self.gRAVE = (np.array([.0]* nsize),np.array([.0]*nsize))
        self.tree = Tree(nsize)
        for i in range(lcount):
            current_features = np.array([False] * nsize, dtype = bool)
            self.iterate(current_features, 0)

    def _attack_one(self, original_label, tokenized_text):
        #The reward function used to evaluate how good current_features is at perturbing input
        def policyvaluefunc(current_features):            
            to_replace = []
            test_input = orig_input
            #Transform each input with current features using given transformation
            for k in range(len(current_features)):
                if current_features[k]:
                    transformed_text_candidates = self.get_transformations(
                        self.transformation,
                        test_input,
                        indices_to_replace=[k])
                    if len(transformed_text_candidates) > 0:
                        rand = np.random.randint(len(transformed_text_candidates))
                        test_input = transformed_text_candidates[rand]

            #Evaluate current features against model
            output = self._call_model([test_input])[0]
            
            if 'combined' in self.rewardvalue:
                prob_exp = torch.exp(output)
                v1 = (prob_exp).data[original_label].clone() 
                (prob_exp).data[original_label] = 0 
                v2 = prob_exp.max().data#[0]
                value = v2 - v1
            elif 'entropy' in self.rewardvalue:
                value = F.nll_loss(output, original_label).data.cpu()[0]
            else:
                value = 1-(torch.exp(output)).data[0, original_label]*2

            value = value.item()
            if self.alltimebest < value:
                self.alltimebest = value
                self.bestfeature = np.copy(current_features)
            return value
                
        self.alltimebest = -1e9
        self.bestfeature = []

        orig_input = tokenized_text
        
        self.runmcts(policyvaluefunc, self.max_words_changed, self.numiter, len(tokenized_text.words))

        new_tokenized_text = tokenized_text

        #Transform each index selected by MCTS using given transformation
        indices = []
        for k in range(len(self.bestfeature)):
            if self.bestfeature[k]:
                transformed_text_candidates = self.get_transformations(
                    self.transformation,
                    new_tokenized_text,
                    indices_to_replace=[k])
                if len(transformed_text_candidates) > 0:
                    rand = np.random.randint(len(transformed_text_candidates))
                    new_tokenized_text = transformed_text_candidates[rand]

        new_output = self._call_model([new_tokenized_text])[0]
        new_text_label = self._call_model([new_tokenized_text])[0].argmax().item()

        if original_label == new_text_label:
            return FailedAttackResult(tokenized_text, original_label)
        else:
            return AttackResult( 
                tokenized_text, 
                new_tokenized_text, 
                original_label,
                new_text_label
            )