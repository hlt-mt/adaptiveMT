#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, getopt, logging
import numpy as np
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import linear_model
import sklearn
import collections
import re
import math
from collections import Counter




'''
tuning class private variables:
1. vector of weights [keep the history]
2. number of updates so far
3. number of iterations
4. learning rate
5. C - mira
6. loss - mira
7. nbest list
8. order of features in list
'''
class Bleu:
	def __init__(self):
	    '''dummy init'''

	def calc_bleu(self, hypo, refe):
	    ref = [refe.strip().split()]
	    hyp = [hypo.strip().split()]
	    stats = [0 for i in xrange(10)]
	    for (r,h) in zip(ref, hyp):
		stats = [sum(scores) for scores in zip(stats, self.bleu_stats(h,r))]
	    
	    return self.bleu(stats)

	def bleu_stats(self, hypothesis, reference):
	    yield len(hypothesis)
            yield len(reference)
	    for n in xrange(1,5):
		s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in xrange(len(hypothesis)+1-n)])
		r_ngrams = Counter([tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n)])
            yield max([sum((s_ngrams & r_ngrams).values())+1, 0])
            yield max([len(hypothesis)+1-n+1, 0])

	def bleu(self, stats):
	    if len(filter(lambda x: x==0, stats)) > 0:
		return 0
	    (c, r) = stats[:2]
	    log_bleu_prec = sum([math.log(float(x)/(y)) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
	    
	    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)


class Tuning:
	def __init__(self, parser):
	    self.C = 1
	    self.time = 0
	    self.numSamples = 15
	    self.learning_rate = 0.1
	    self.learning_rate = float(parser.get('tuning', 'learningrate'))
	    self.weights_init=[]
	    self.sigma_init=[]
	    self.sparseSigma=[]
	    self.labels=[]
	    self.numCoreFeatures=0
	    self.initFeaturesDict=[]
	    self.initFeaturesMap={}
	    self.active_sparse_feats={}
	    self.sparseFeaturesMap={}
	    self.sparseFeatures=[]

	def annotate(self):
	    weig_ann = "<weight-overwrite weights=\"";
	    weights = self.weights_init
	    offset=0
	    values=[]
	    curr_values=[]
	    for name, values in self.initFeaturesDict:
		curr_values = weights[offset:offset+len(values)]
		s =  ' '.join(str(x) for x in curr_values)
		weig_ann = weig_ann + name + "= " + s + " "
		offset += len(values)

	    for k in self.sparseFeaturesMap:
		sW = self.getSparseFeatWeightbyName(k)
		weig_ann = weig_ann + k + "= " + str(sW) + " "

	    weig_ann = weig_ann + "\"></weight-overwrite>";
	    logging.info("annotation : "+weig_ann)
	    
	    return weig_ann

	# read initial feature list
	def read_init_features(self, feats):
	    features = feats.split()
	    values=[]
	    self.initFeaturesDict = []
	    prevfeat=""
	    for f in features:
		if '=' in f:
		    if len(values) == 0:
			prevfeat = f[:-1]
			continue
		    self.initFeaturesDict.append((prevfeat, values))
		    self.initFeaturesMap[prevfeat]=1
		    prevfeat = f[:-1]
		    values=[]
		else:
		    self.weights_init.append(float(f))
		    self.sigma_init.append(0)
		    values.append(float(f))
		    self.numCoreFeatures += 1
	    
	    self.initFeaturesDict.append((prevfeat, values))
	    self.initFeaturesMap[prevfeat]=1
	   

	def read_features(self, feats):
	    features = feats.split() # feature string from nbest list
	    values=[]	# contains temporary feature values
	    dense_feats = {} # maps dense features to a list of values
	    sparse_feats = {}	# contains active sparse features in the nbest candidate
	    prevfeat=""
	    count=0
	    for f in features:
		if '=' in f:
		    if len(values) == 0:
			prevfeat = f[:-1]
			continue
		    ## check if the feature is a dense or a sparse feature
		    if not prevfeat in self.initFeaturesMap:
			sparse_id = self.getSparseFeatID(prevfeat)
		    ## check if sparse feature exists
			if sparse_id == -1 :
			    sparse_id = self.setSparseFeatWeight(prevfeat, 0)
			sparse_feats[sparse_id] = values[0]
			self.active_sparse_feats[sparse_id]=1
		    else:
			dense_feats[prevfeat] = values
		    prevfeat = f[:-1]
		    values=[]
		else:
		    values.append(float(f))
		    count += 1

# for the last feature
	    if not prevfeat in self.initFeaturesMap:
		sparse_id = self.getSparseFeatID(prevfeat)
	    ## check if sparse feature exists
		if sparse_id == -1 :
		    sparse_id = self.setSparseFeatWeight(prevfeat, 0)
		sparse_feats[sparse_id] = values[0]
		self.active_sparse_feats[sparse_id]=1
	    else:
		dense_feats[prevfeat] = values

	    features=[]
	    for k,v in self.initFeaturesDict:
		features.extend(dense_feats[k])

	    return features, sparse_feats
	
	def getSparseFeatID(self, name):
	    if name in self.sparseFeaturesMap:
		return self.sparseFeaturesMap[name]
	    return -1

	def getSparseFeatWeightbyName(self, name):
	    score=-1
	    if name in self.sparseFeaturesMap:
		score = self.sparseFeatures[self.sparseFeaturesMap[name]]
	    return score

	def getSparseFeatWeightbyID(self, id):
	    score=-1
	    if id < len(self.sparseFeatures):
		score = self.sparseFeatures[id]
	    return score

	def setSparseFeatWeightbyID(self, id, val):
	    if id < len(self.sparseFeatures):
		self.sparseFeatures[id] = val
		return 1
	    return -1

	def setSparseFeatSigmabyID(self, id, val):
	    if id < len(self.sparseSigma):
		self.sparseSigma[id] = val
		return 1
	    return -1

	def setSparseFeatWeight(self, name, val):
	    if name in self.sparseFeaturesMap:
		id = self.sparseFeaturesMap[name]
		self.sparseFeatures[id] = val
		return id
	    else:
		self.sparseFeatures.append(val)
		self.sparseSigma.append(0)
		self.sparseFeaturesMap[name] = len(self.sparseFeatures)-1
		return len(self.sparseFeatures)-1
	
	def getLearningRate(self):
	    return self.learning_rate * 2 / (1 + e**(self.time))

	def pro_update(self, label, features, weights_init, sigma_init):
	    # sample derivation pairs R times
	    if len(label) <= self.numSamples*3:
		return
	    a = np.random.normal(loc=len(label), scale=len(label)/3, size=(self.numSamples,1)).tolist()
	    b = np.random.normal(loc=len(label), scale=len(label)/3, size=(self.numSamples,1)).tolist()
	    samples = zip(a,b)
	    # compute log loss from R samples
	    loss = 0
	    weights = np.zeros(len(weights_init)).tolist()
	    sigma = np.zeros(len(sigma_init)).tolist()
	    for i in xrange(len(samples)):
		idx_x = int(samples[i][0][0])
		idx_y = int(samples[i][1][0])
		while idx_x >= len(label):
		    idx_x = int(np.random.normal(loc=len(label), scale=len(label)/3))
		while idx_y >= len(label):
		    idx_y = int(np.random.normal(loc=len(label), scale=len(label)/3))
		G_x = label[idx_x]
		G_y = label[idx_y]
		sign = 1
		if G_x > G_y:
		    for j in xrange(len(features[0])):
			diff = features[idx_x][j] - features[idx_y][j]
			loss = -1 * diff * math.exp(-1 * weights_init[j] * diff)
			sigma[j] += math.pow(loss, 2)
			if sigma_init[j] > 0:
			    weights[j] -= self.learning_rate * loss * math.sqrt(1/sigma_init[j])
		elif G_x < G_y:
		    for j in xrange(len(features[0])):
			diff = features[idx_y][j] - features[idx_x][j]
			sigma[j] += math.pow(loss, 2)
			loss = -1 * diff * math.exp(-1 * weights_init[j] * diff)
			if sigma_init[j] > 0:
			    weights[j] -= self.learning_rate * loss * math.sqrt(1/sigma_init[j])

	    for i in xrange(len(weights)):
		weights_init[i] += float(weights[i]/len(samples))
	    for i in xrange(len(sigma)):
		sigma_init[i] += float(sigma[i]/len(samples))

	    return weights_init, sigma_init

	    
	def optimize(self, nbest, pe):
	    if len(nbest) < 5:
		return ""
	    self.time += 1
	    count=0
	    self.features=[]
	    self.model_scores=[]
	    self.labels=[]
	    Bleu_obj = Bleu()
	    self.active_sparse_feats={}
	    for N in nbest:
		temp = N
		temp = re.sub(r"^[^\|]+\|\|\|\s*","",temp)
		feats = re.sub(r"^[^\|]+\|\|\|\s*","",temp)
		# model scores is a list with size N
		self.model_scores.append(float(re.sub(r"^[^\|]+\|\|\|\s*","",feats)))
		feats = re.sub(r"\s*\|\|\|.+$","",feats)

		# crucial step
		dense_feats, sparse_feats = self.read_features(feats)
		
		# features is a list of list size N x d
		self.features.append((list(dense_feats), sparse_feats.copy()))
		nbest_ = re.sub(r"\s*\|\|\|.+$","",temp)
		bleuscore = Bleu_obj.calc_bleu(nbest_, pe)
		self.labels.append(bleuscore)		    # label is a list with size N
		count += 1

	    self.optimize_(count)
	    return self.annotate()
	    

	def optimize_(self, count):
	    idx=self.labels.index(max(self.labels))
	    feats=[]
	    y = np.asarray(self.labels, dtype=np.float64)
#size of sparse vector 
	    size = len(self.active_sparse_feats)
#	    logging.info("Size of sparse features : "+str(size)) 
# create an index for sparse vector
	    index_sparse=[]
	    for k,v in self.active_sparse_feats.iteritems():
		index_sparse.append(k)

# create a sparse feature vector and corresponding sparse weight vector
	    for i in xrange(count):
		temp_feat_list = list(self.features[i][0])
		
		temp_sparse_dict=self.features[i][1]
		for j in xrange(size):
# append the value of the sparse feature with index = index_sparse[j]
		    if index_sparse[j] in temp_sparse_dict:
			temp_feat_list.append(temp_sparse_dict[index_sparse[j]])
		    else:
			temp_feat_list.append(0.)
		feats.append(temp_feat_list)
	    X = np.asarray(feats, dtype=np.float64)
	    
# create weight and sigma vector from dense and sparse features
	    W = list(self.weights_init)
	    S = list(self.sigma_init)
# extend W with sparse features
	    for i in xrange(size):
		W.append(self.sparseFeatures[index_sparse[i]])
		S.append(self.sparseSigma[index_sparse[i]])
	    
#	    logging.info("Best Bleu :"+ str(y[0,]))
#	    logging.info("1-Best Features :"+ str(X[0,]))
#	    logging.info("Index :"+ str(idx))
#	    logging.info("Oracle Bleu :"+ str(y[idx,]))
#	    logging.info("Oracle Features :"+ str(X[idx,]))
	    Feat = X.copy()
	    Bleu = y.copy()
	    
#	    s =  ' '.join(str(x) for x in index_sparse)
#	    logging.info("Sparse Index : "+s) 
#	    s =  ' '.join(str(x) for x in W)
#	    logging.info("Weights : "+s) 
#	    s =  ' '.join(str(x) for x in S)
#	    logging.info("Sigma : "+s) 
# Pairwise ranking approach with the logistic loss : same as Green et. al. 2013
	    weights, sigma = self.pro_update(Bleu, Feat, W, S)

# set weights and sigma for dense and sparse features
	    self.weights_init = weights[:self.numCoreFeatures]
	    self.sigma_init = sigma[:self.numCoreFeatures]
#	    s =  ' '.join(str(x) for x in self.weights_init)
#	    logging.info("Weights : "+s) 
#	    s =  ' '.join(str(x) for x in self.sigma_init)
#	    logging.info("Sigma : "+s) 
	    for i in xrange(self.numCoreFeatures, len(weights)):
#		logging.info("Setting weights for "+str(index_sparse[i-self.numCoreFeatures])+" --> "+str(weights[i]))
#		logging.info("Setting sigma for "+str(index_sparse[i-self.numCoreFeatures])+" --> "+str(sigma[i]))
		self.setSparseFeatWeightbyID(index_sparse[i-self.numCoreFeatures], weights[i])
		self.setSparseFeatSigmabyID(index_sparse[i-self.numCoreFeatures], sigma[i])


