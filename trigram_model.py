import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""
# generates series of tokens by reading from the corpusfile and matching individual words with the lexicon
def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split() # split line string into list of words
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

# return the set of all words which appear more than once in the corpus
def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    ls = []
    j = 0
    i = j - n + 1
    if (n == 1):
        ls.append(('START',))
    while (j <= len(sequence)):
        newItem = []
        for k in range(i, j+1):
            if (k < 0):
                newItem.append('START')
            elif (k >= len(sequence)):
                newItem.append('STOP')
            else: 
                newItem.append(sequence[k])
        ls.append(tuple(newItem))
        i+=1
        j+=1
    
    return ls


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.get_total_counts()

    def print_unigrams(self):
        for key in self.unigramcounts: 
            print(key, "->", self.unigramcounts[key])


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
    
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here 
        for sentence in corpus:
            for n in range(1,4):
                ls = get_ngrams(sentence, n)
                for tup in ls:
                    if (n == 1):
                        self.unigramcounts[tup] += 1 
                    elif (n == 2):
                        self.bigramcounts[tup] += 1
                    else:
                        self.trigramcounts[tup] += 1
        return 0

    # returns the total number of unigrams, bigrams, and trigrams
    def get_total_counts(self):
        self.unigramtotal = 0
        self.bigramtotal = 0
        self.trigramtotal = 0
        for _, val in self.unigramcounts.items():
            self.unigramtotal += val
        for _, val in self.bigramcounts.items():
            self.bigramtotal += val
        for _, val in self.trigramcounts.items():
            self.trigramtotal += val

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        p(w | u, v) = p(u, v, w) / p(u, v)
        """
        return float(self.trigramcounts[trigram]) / float(self.trigramtotal)

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        p(w | v) = p(w, v) / p(v)
        """
        
        return float(self.bigramcounts[bigram]) / float(self.bigramtotal) 
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # print("unigram_counts[{name}] = {count}".format(name=unigram, count=self.unigramcounts[unigram]))
        # print("unigram_total = {}".format(self.unigramtotal))

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return float(self.unigramcounts[unigram]) / float(self.unigramtotal)

    # Helper function: Returns all trigrams with the first two words 
    def search_context(self, bigram):
        res = []
        for key, _ in self.trigramcounts.items():
            if (key[0] == bigram[0] and key[1] == bigram[1]):
                res.append(key)
        return res
    
    # Helper function: Returns the most likely next word given a bigram context
    def get_next_word(self, trigram_ls):

        """
        the_entry = None
        max_prob = 0
        for entry in trigram_ls:
            newP = self.raw_trigram_probability(entry)
            if (newP > max_prob):
                max_prob = newP
                the_entry = entry
        return the_entry[2]     
        """
        if (len(trigram_ls) == 0):
            return "a"
        
        index = random.randint(0, len(trigram_ls)-1)
        return trigram_ls[index][2] 
        
    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        
        result = ["START", "START"]
        
        b = 0 # position of the first word in bigram
        while (b < t-2 and result[b+1] != "STOP"):
            cur_bigram = result[b:b+2]
            ls = self.search_context(cur_bigram)
            result.append(self.get_next_word(ls))
            b+=1
        del result[0:2] # remove START's
        if (result[len(result)-1] != 'STOP'):
            result.append('STOP')
    
        print(result)
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        """
        Given a trigram (u, v, w)
        r: P(w)
        s: P(v)
        x: P((u, v))
        y: P((v, w))
        z: P((u, v, w))
        """
        r = self.raw_unigram_probability(tuple([trigram[2]]))
        s = self.raw_unigram_probability(tuple([trigram[1]]))
        x = self.raw_bigram_probability(trigram[0:2])
        y = self.raw_bigram_probability(trigram[1:3])
        z = self.raw_trigram_probability(trigram)
        
        #print("x is {}", x)
        #print("s is {}", s)

        if (r == 0):
            return 0
        elif (y == 0 or s == 0):
            return lambda3 * r
        elif (x == 0):
            return lambda2 * (float(y) / s)
    
        return lambda1 * (float(z) / float(x)) + lambda2 * (float(y) / float(s)) + lambda3 * r
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        probs = [self.smoothed_trigram_probability(i) for i in get_ngrams(sentence, 3)]
        x = sum(math.log(prob, 2) for prob in probs) 
        return x

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        res = 0
        M = 0
        for line in corpus:
            res += self.sentence_logprob(line)
            M += len(line)
        
        return math.pow(2, -res / M)

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_wrong = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if (pp <= pp_wrong):
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_wrong = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if (pp <= pp_wrong):
                correct += 1
            total += 1

        return float(correct) / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    model.generate_sentence()

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity: 
    #dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    #print(acc)

