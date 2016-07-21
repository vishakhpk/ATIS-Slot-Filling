import theano
import numpy
import os
import gzip
import six.moves.cPickle as Pickle
import urllib
import random
import time
import sys
import stat
import subprocess
from os import chmod
from theano import tensor as T
from collections import OrderedDict
from os.path import isfile

PREFIX = os.getenv('ATISDATA', '')

class RNN(object):

    def __init__(self, nHidden, nLabels, vocabularySize, dmnEmbedding, contextWindowSize):
        self.embeddings = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (vocabularySize + 1, dmnEmbedding)).astype(theano.config.floatX))

        self.weightsInput = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (dmnEmbedding * contextWindowSize, nHidden)).astype(theano.config.floatX))

        self.weightsHidden = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nHidden, nHidden)).astype(theano.config.floatX))

        self.weightsOutput = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,  (nHidden, nLabels)).astype(theano.config.floatX))

        self.biasesHidden = theano.shared(numpy.zeros(nHidden, dtype=theano.config.floatX))

        self.biasesOutput = theano.shared(numpy.zeros(nLabels, dtype=theano.config.floatX))

        self.initialState = theano.shared(numpy.zeros(nHidden, dtype=theano.config.floatX))

        self.params = [self.embeddings, self.weightsInput, self.weightsHidden, self.weightsOutput, self.biasesHidden, self.biasesOutput, self.initialState]
        self.names = ['embeddings', 'weightsInput', 'weightsHidden', 'weightsOutput', 'biasesHidden', 'biasesOutput', 'initialState']
        idxs = T.imatrix()
        x = self.embeddings[idxs].reshape((idxs.shape[0], dmnEmbedding * contextWindowSize))
        y = T.iscalar('y')

        def recurrence(x_t, pastState):
            currentState = T.nnet.sigmoid(T.dot(x_t, self.weightsInput) + T.dot(pastState, self.weightsHidden) + self.biasesHidden)
            predictNext = T.nnet.softmax(T.dot(currentState, self.weightsOutput) + self.biasesOutput)
            return [currentState, predictNext]

        [nextState, predictNext], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.initialState, None], n_steps=x.shape[0])

        predictYGivenXLastword = predictNext[-1, 0, :]
        predictYGivenXSentence = predictNext[:, 0, :]
        yPrediction = T.argmax(predictYGivenXSentence, axis=1)

        alpha = T.scalar('a')
        negativeLogLikelihood = -T.log(predictYGivenXLastword)[y]
        gradients = T.grad(negativeLogLikelihood, self.params)
        updates = OrderedDict((p, p - alpha * g) for p, g in zip(self.params, gradients))

        self.classify = theano.function(inputs=[idxs], outputs=yPrediction)

        self.train = theano.function(inputs=[idxs, y, alpha], outputs=negativeLogLikelihood, updates=updates)

        self.normalize = theano.function(inputs=[],
                                         updates={self.embeddings: self.embeddings / T.sqrt((self.embeddings ** 2).sum(axis=1)).dimshuffle(0,'x')})

def loadDataset():
    filename = PREFIX + 'atis.fold3.pkl.gz'
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = Pickle.load(f)
    return train_set, valid_set, test_set, dicts


def shuffle(lol, seed):
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def minibatch(l, bs):
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    return out


def contextwin(l, win):
    l = list(l)
    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]
    return out


def conlleval(p, g, w, filename):
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)


def get_perf(filename):
    _conlleval = PREFIX + 'conlleval.pl'
    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[3][:-2])
    recall = float(out[5][:-2])
    f1score = float(out[7])

    return {'p': precision, 'r': recall, 'f1': f1score}

if __name__ == '__main__':
    
    noOfFolds=3
    seed=345
    nHidden=100
    embDimension=100
    contextWindowSize=7
    alpha=0.0627142536696559
    nEpochs=50
    nBpttSteps=9
    verbose=1
    
    trainSet, validationSet, testSet, dic = loadDataset()
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    trainSentences, trainExtra, trainSetY = trainSet
    validationSentences, validationExtra, validationSetY = validationSet
    testSentences,  testExtra,  testSetY  = testSet

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(trainSentences)

    numpy.random.seed(seed)
    random.seed(seed)
    rnn = RNN(nHidden=nHidden,
                nLabels=nclasses,
                vocabularySize=vocsize,
                dmnEmbedding=embDimension,
                contextWindowSize=contextWindowSize)

    best_f1 = -numpy.inf
    
    s = {}

    for e in xrange(nEpochs):
        shuffle([trainSentences, trainExtra, trainSetY], seed)
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(trainSentences[i], contextWindowSize)
            words = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(cwords, nBpttSteps))
            labels = trainSetY[i]
            for word_batch, label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, alpha)
                rnn.normalize()
            sys.stdout.flush()

        predictionsTest = [map(lambda x: idx2label[x], rnn.classify(numpy.asarray(contextwin(x, contextWindowSize)).astype('int32'))) for x in testSentences]

        testY = [map(lambda x: idx2label[x], y) for y in testSetY]

        wordsTest = [map(lambda x: idx2word[x], w) for w in testSentences]

        predictionsValidation = [map(lambda x: idx2label[x], rnn.classify(numpy.asarray(contextwin(x, contextWindowSize)).astype('int32'))) for x in validationSentences]

        validationY = [map(lambda x: idx2label[x], y) for y in validationSetY]

        wordsValidation = [map(lambda x: idx2word[x], w) for w in validationSentences]

        testResult = conlleval(predictionsTest, testY, wordsTest, 'current.test.txt')

        validationResult = conlleval(predictionsValidation, validationY, wordsValidation, 'current.valid.txt')

        if validationResult['f1'] > best_f1:
            best_f1 = validationResult['f1']
            print 'NEW BEST: epoch', e, 'valid F1', validationResult['f1'], 'best test F1', testResult['f1'], ' ' * 20
            s['vf1'], s['vp'], s['vr'] = validationResult['f1'], validationResult['p'], validationResult['r']
            s['tf1'], s['tp'], s['tr'] = testResult['f1'], testResult['p'], testResult['r']
            s['be'] = e

    print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model'
