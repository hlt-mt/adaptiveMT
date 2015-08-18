#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, getopt, logging
import numpy as np
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import linear_model
import sklearn
import nltk
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
	    self.numIter=1
	    self.C = 1
	    self.time = 0
	    self.numSamples = 15
	    self.learning_rate = 0.1
	    self.learning_rate = float(parser.get('tuning', 'learningrate'))
	    self.numIter = int(parser.get('tuning', 'iterations'))
	    self.C = float(parser.get('tuning', 'C'))
	    self.loss = parser.get('tuning', 'loss')
	    self.numUpdates = 0
	    self.alpha = 0.01
	    self.slack = 0.001
	    self.features=[]
	    self.model_scores=[]
	    self.weights_init=[]
	    self.labels=[]
	    self.sigma = np.zeros(1000).tolist()
	    self.initFeaturesDict=[]
	    self.algorithm = parser.get('tuning', 'algorithm')
	    if self.algorithm == "SGD":
		self.classifier = sklearn.linear_model.SGDRegressor(loss='huber', penalty='elasticnet', n_iter=10, warm_start=True, average=True, fit_intercept=True, alpha=0.00001, random_state=1, shuffle=True)
	    elif self.algorithm == "PA":
		self.classifier = sklearn.linear_model.PassiveAggressiveRegressor(loss='epsilon_insensitive', C=1, warm_start=True, epsilon=0.0001, n_iter=100, fit_intercept=True)
	    elif self.algorithm == "AdaGrad":
		self.classifier = AdaGrad(loss="hinge", C=1, epsilon=0.001, n_iter=100, fit_intercept=True)
#sklearn.linear_model.PassiveAggressiveRegressor(loss='epsilon_insensitive', C=1, warm_start=True, epsilon=0.0001, n_iter=100, fit_intercept=True)



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

	    weig_ann = weig_ann + "\"></weight-overwrite>";
	    logging.info("annotation : "+weig_ann)
	    
	    return weig_ann

	def read_1best(self, nbest):
	    self.firstbest = nbest[0]
            self.firstbest = re.sub(r"^[^\|]+\|\|\|\s*","",firstbest)
            self.firstbest = re.sub(r"\s*\|\|\|.+$","",firstbest)

#	def bleu(self, hyp, ref):
#	    hyp_grams = hyp.split()
#	    ref_grams = ref.split()
#	    weights = [0.25, 0.25, 0.25, 0.25]
#	    total = nltk.align.bleu_score.bleu(hyp_grams, ref_grams, weights)
#	    return total

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
		    prevfeat = f[:-1]
		    values=[]
		else:
		    self.weights_init.append(float(f))
		    values.append(float(f))
	    
	    self.initFeaturesDict.append((prevfeat, values))
	   

	def read_features(self, feats):
	    features = feats.split()
	    values=[]
	    list_feats = {}
	    prevfeat=""
	    for f in features:
		if '=' in f:
		    if len(values) == 0:
			prevfeat = f[:-1]
			continue
		    list_feats[prevfeat] = values
		    prevfeat = f[:-1]
		    values=[]
		else:
		    values.append(float(f))

	    list_feats[prevfeat] = values

	    features=[]
	    for k,v in self.initFeaturesDict:
		features.extend(list_feats[k])

	    return features
	
	def getFeatId(self, name):
	    if name in self.initFeaturesDict:
		return self.initFeaturesDict[name]
	    # make new feature ID
	    featureList.append(name)
	    return len(featureList)
	
	def getLearningRate(self):
	    return self.learning_rate * 2 / (1 + e**(self.time))

	def pro_update(self, prediction, label, features):
	    # sample derivation pairs R times
	    if len(label) <= self.numSamples*3:
		return
	    a = np.random.normal(loc=len(label), scale=len(label)/3, size=(self.numSamples,1)).tolist()
	    b = np.random.normal(loc=len(label), scale=len(label)/3, size=(self.numSamples,1)).tolist()
	    samples = zip(a,b)
	    # compute log loss from R samples
	    loss = 0
	    weights = np.zeros(len(self.weights_init)).tolist()
	    self.sigma = self.sigma[:len(self.weights_init)]
	    sigma = np.zeros(len(self.sigma)).tolist()
	    for i in xrange(len(samples)):
		idx_x = int(samples[i][0][0])
		idx_y = int(samples[i][1][0])
		while idx_x >= len(label):
		    idx_x = int(np.random.normal(loc=len(label), scale=len(label)/3))
		while idx_y >= len(label):
		    idx_y = int(np.random.normal(loc=len(label), scale=len(label)/3))
		G_x = label[idx_x]
		G_y = label[idx_y]
		M_x = prediction[idx_x]
		M_y = prediction[idx_y]
		sign = 1
		if G_x > G_y:
		    for j in xrange(len(features[0])):
			diff = features[idx_x][j] - features[idx_y][j]
			#loss = sign*math.log(1 + math.exp(-1 * self.weights_init[j] * diff))
			loss = -1 * diff * math.exp(-1 * self.weights_init[j] * diff)
			sigma[j] += math.pow(loss, 2)
			if self.sigma[j] > 0:
			    weights[j] -= self.learning_rate * loss * math.sqrt(1/self.sigma[j])
		elif G_x < G_y:
		    for j in xrange(len(features[0])):
			diff = features[idx_y][j] - features[idx_x][j]
			#loss = sign*math.log(1 + math.exp(-1 * self.weights_init[j] * diff))
			sigma[j] += math.pow(loss, 2)
			loss = -1 * diff * math.exp(-1 * self.weights_init[j] * diff)
			if self.sigma[j] > 0:
			    weights[j] -= self.learning_rate * loss * math.sqrt(1/self.sigma[j])

	    s =  ' '.join(str(x) for x in weights)
	    logging.info("AvgUpdate : "+s)
	    s =  ' '.join(str(x) for x in self.sigma)
	    logging.info("Sigma : "+s)
	    for i in xrange(len(weights)):
		self.weights_init[i] += float(weights[i]/len(samples))
#		if self.weights_init[i] - self.l1lambda * self.learning_rate > 0:
#		    self.weights_init[i] -= self.l1lambda * self.learning_rate	
#		else:
#		    self.weights_init[i]=0
	    for i in xrange(len(sigma)):
		self.sigma[i] += float(sigma[i]/len(samples))


	    # using FOBOS framework 
	    # regularize weights
	    
	def optimize(self, nbest, pe):
	    self.time += 1
	    count=0
	    self.features=[]
	    self.model_scores=[]
	    self.labels=[]
	    Bleu_obj = Bleu()
	    for N in nbest:
		temp = N
		temp = re.sub(r"^[^\|]+\|\|\|\s*","",temp)
		feats = re.sub(r"^[^\|]+\|\|\|\s*","",temp)

		# model scores is a list with size N
		self.model_scores.append(float(re.sub(r"^[^\|]+\|\|\|\s*","",feats)))

		feats = re.sub(r"\s*\|\|\|.+$","",feats)
		feat = self.read_features(feats)
		self.features.append(feat)		    # features is a list of list size N x d
		nbest_ = re.sub(r"\s*\|\|\|.+$","",temp)
		bleuscore = Bleu_obj.calc_bleu(nbest_, pe)
#		logging.info("NBEST : "+nbest_+" BLEU : "+str(bleuscore))
		self.labels.append(bleuscore)		    # label is a list with size N
		count += 1

	    self.optimize_(count)
	    return self.annotate()
	    

	def optimize_(self, count):
	    idx=self.labels.index(max(self.labels))
	    
	    w = np.asarray(self.model_scores, dtype=np.float64)
	    X = np.asarray(self.features, dtype=np.float64)
	    y = np.asarray(self.labels, dtype=np.float64)
	    
	    # create a \delta feature array
	    oracleFeat = X[idx,]
	    oracleBleu = y[idx,]
	    oracleModelScore = w[idx,]
	    logging.info("Oracle Bleu :"+ str(oracleBleu))
	    logging.info("Index :"+ str(idx))
	    logging.info("1-Best Features :"+ str(X[0,]))
	    logging.info("Oracle Features :"+ str(X[idx,]))
	    deltaFeat = X.copy()
	    deltaBleu = y.copy()
	    deltaModelScores = w.copy()
	    numUpdates=0
	    maxUpdates=10
	    s =  ' '.join(str(x) for x in self.weights_init)
	    logging.info("Weights : "+s) 
	    coefs = np.asarray(self.weights_init).reshape(1, X.shape[1])

# Pairwise ranking approach with the logistic loss : same as Green et. al. 2013
	    self.pro_update(deltaModelScores, deltaBleu, deltaFeat)
	    s =  ' '.join(str(x) for x in self.weights_init)
	    logging.info("Weights : "+s) 



#######################################################################################
#	    loss = calc_loss(deltaModelScores, deltaBleu, deltaFeat)
	    

#	    s =  ' '.join(str(x) for x in y)
#	    logging.info("Bleu : "+s) 
#	    s =  ' '.join(str(x) for x in deltaModelScores)
#	    logging.info("Delta Model Scores : "+s) 
#	    s =  ' '.join(str(x) for x in loss)
#	    logging.info("Loss : "+s) 
#	    self.classifier.fit(deltaFeat, loss, coef_init=coefs)
#	    self.weights_init= np.transpose(self.classifier.coef_).tolist()


# Code I wrote for updating weights .. no theoritical base
#		loss = deltaModelScores[i] - deltaBleu[i]
#		logging.info("Model Score diff : "+str(deltaModelScores[i]) + "\t BleuScoreDiff : "+str(deltaBleu[i])+"\tloss : "+str(loss))
#		if loss > self.slack and numUpdates < maxUpdates:
#		    numUpdates +=1
#		    for j in xrange(X.shape[1]):
#			logging.info("Updating for candidate "+str(i)+ " feature " + str(j) +" with a loss of "+ str(deltaFeat[i,j]) ) 
#			direction = -1 if deltaFeat[i,j] < 0 else 1 
#			self.weights_init[j] -= self.alpha * self.slack * direction


