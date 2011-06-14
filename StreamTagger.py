#####
# StreamTagger.py: For integration with SharePoint 2010 Social Tagging
# Run with Python 2.6.6, nltk 2, numpy 1.6, PyYAML 3.
# ex.: c:\python26\python c:\bin\StreamTagger < stream
# ARGUMENTS: (STDIN) rawtext
# Output is tab delimited which can be directly imported into SQL Server or Excel.
# TWEEKS section contains modifyable performance variables.
#####
import sys, nltk, os, re, numpy
from nltk.tokenize import *

#####
# TWEEKS
# Change any of these to alter the performance of the system
#####
PASSFAIL = 35.0 # Sets cutoff for RSD of file. To process more files, reduce. Minimum is around 25.
NGRAM_Cutoff = 15 # set to -1 for no cutoff
BIGRAM_Cutoff = 15 # set to -1 for no cutoff
UNIGRAM_Cutoff = 10 # set to -1 for no cutoff

def POS_Output(sentences):
    # Useful for debugging to see what the parts of speech are after the POS tagger has
    # worked on the tokenized sentences.
    for sent in sentences:
        pt_sent = []
        pos_sent = []
        for tword in sent:
            pt_sent.append(tword[0])
            pos_sent.append(tword[1])
        print "\t".join(pt_sent) + "\n" + "\t".join(pos_sent)

patterns = r""" # define tag patterns
            NP: {<JJ>*<NN>+} # chunk adjectives and noun
                {<JJ>*<NNS>+}
                {<NN><JJ>*(<NN>+|<NNS>+)}
                {<JJ>*<NNS>+} # chunk adjectives and noun
                {<NN>+} # chunk consecutive nouns
                {<NNP>+} # chunk sequences of proper nouns
                {<JJ><CC><JJ><NN>} # adjective conjunction adjective noun
                {<NN><NNS>}
                {<NNS><NNS>}
                {<VBN>*<NN>+}
          """
      
def ConceptMap(sentences):
# Given plaintext it produces a concept map, returned as a hash table
    NPChunker = nltk.RegexpParser(patterns) #create chunk parser
    NPgroup = ''  #an individual tracked item
    NPlist = {} # Concept Map as a hash table
    for sent in sentences:
      for chunk in NPChunker.parse(sent):
        if type(chunk) != tuple: #non-NP items remain as tuples, chunks are tree objects
          for leaf in chunk.leaves():
            if len(NPgroup) > 0:
              NPgroup += ' ' + leaf[0]
            else:
              NPgroup += leaf[0]      
          if len(NPgroup) == 0: # sometimes an empty group comes up, we just ignore it.
              continue
          if NPgroup[0].isalpha() and len(NPgroup) > 2:  #filter oddities
            if NPgroup in NPlist:
              NPlist[NPgroup] += 1
            else:
              NPlist[NPgroup] = 1
          NPgroup = ''  #clear item and look for the next one
    return NPlist
      
#####
# StoreConceptMap
# This is what sends the output to STDOUT after storing the information locally in NPlist
#####
def StoreConceptMap(ConceptMap):
    UNIGRAM = []
    BIGRAM = []
    NGRAM = []
    global recordnum
    NPlist = ConceptMap
    # list is sorted by Frequency highest to lowest
    for key, value in sorted(NPlist.items(), key=lambda(k,v):(v,k), reverse=True):
        key = key.strip() # clean up extra start and end spaces
        key = re.sub(r'  +',r' ',key) # condense multiple internal spaces to single spaces
        key = key.lower() # flatten case so that terms are more likely to associate to the same tag
        kc = key.count(' ')
        if kc == 0:
            UNIGRAM.append([key, value])
        elif kc == 1:
            BIGRAM.append([key, value])
        elif kc > 6:
            continue # an NGram this big is most likely junk
        else:
            NGRAM.append([key, value])
    # calculate RSD - if file fails do not process
    termcount = len(NGRAM) + len(BIGRAM) + len(UNIGRAM)
    totalfreq = 0
    averagefreq = stddevfreq = 0.0
    freqarray = []
    for gram, freq in NGRAM:
        totalfreq += freq
        freqarray.append(freq)
    for gram, freq in BIGRAM:
        totalfreq += freq
        freqarray.append(freq)
    for gram, freq in UNIGRAM:
        totalfreq += freq
        freqarray.append(freq)
    averagefreq = float(totalfreq)/float(termcount)
    stddevfreq = numpy.std(freqarray)
    RSD = (stddevfreq/averagefreq)*100
    if RSD >= PASSFAIL:        
        # Limits are placed on maximum output
        NGC = BGC = UGC = 10000
        if NGRAM_Cutoff != -1:
            NGC = NGRAM_Cutoff -1
        if BIGRAM_Cutoff != -1:
            BGC = BIGRAM_Cutoff -1
        if UNIGRAM_Cutoff != -1:
            UGC = UNIGRAM_Cutoff -1
            
        for gram, freq in NGRAM[0:NGC]:
            if freq > 1:
                print '{0}\t'.format(gram),
        for gram, freq in BIGRAM[0:BGC]:
            if freq > 1:
                print '{0}\t'.format(gram),
        for gram, freq in UNIGRAM[0:UGC]:
            if freq > 1:
                print '{0}\t'.format(gram),
    else:
        quit(-1)


def loadSignalPattern():
    # Open STDIN and feed plaintext
    plaintext = sys.stdin.read()
    sentences = nltk.sent_tokenize(plaintext) # NLTK default sentence segmenter
    tokenizer = PunktWordTokenizer()
    sentences = [tokenizer.tokenize(sent) for sent in sentences] #wordpunkt tokenizer
                
    # Remove obviously garbage scentences - some contain many, very short tokens. For
    # example a font reference will contain every letter of the alphabet at one word
    # each. Determine if the sentence is garbarge by examining it's length and average
    # token length.                
    for i in range(len(sentences)-1,-1,-1):
        tokensizes = 0
        if len(sentences[i]) < 6:
            continue
        for token in sentences[i]:
            tokensizes += len(token)
        if tokensizes/len(sentences[i]) < 3:
            sentences.pop(i)
                
    # The Punkt tokenizer appends the final period to the final word of a sentence
    # and we don't want that so we have to clean that up.
    for si in range(len(sentences)-1,-1,-1):
        sent = sentences[si]
        # This If code cleans up single toke sentences. This is the best place to toss them out.
        if len(sent) == 1:
            sentences.pop(si)
            continue
        word = sent[len(sent)-1]
        if word[len(word)-1] == '.':
            sent[len(sent)-1] = word[:len(word)-1] #chop off the period
            sent.append(".") # create it's own token
                
    # Repair transcription errors and remove garbage
    for si in range(len(sentences)-1,-1,-1):
        sent = sentences[si]
        try:
            for i in range(len(sent)-1, -1, -1): #since we might be popping items we go backwards to keep the index intact
                if sent[i].find('\x92') > -1: # repair apostrophes
                    sent[i] = sent[i].replace('\x92',"\'")
                if sent[i].isalnum() != True: #test for potential garbage characters
                    for j in range(len(sent[i])-1,-1,-1): #scour each token for junk characters
                        try:
                            if "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.!?,;:&@[]{}$%<>\\~'".find(sent[i][j]) == -1:
                                sent[i] = sent[i].replace(sent[i][j],' ')
                        except:
                            pass
        except:
            raise
        sent[0] = sent[0].capitalize() # useful for the parser to just make sure this is set
                
    sentences = [nltk.pos_tag(sent) for sent in sentences] # NLTK POS tagger
    #outpos(sentences) #diagnostic
    StoreConceptMap(ConceptMap(sentences))
    return(1)

# outpos produces a tab delimited Part of Speech breakdown of the input.
# This is useful for adjusting the grammar patterns for the chunker.
# It can be given any proper sentences group with POS tagging from
# nlkt.pos_tag
def outpos(sentences):
    for sent in sentences:
        wordlist = []
        poslist = []
        for tword in sent:
            wordlist.append(tword[0])
            poslist.append(tword[1])
        print "\t".join(wordlist)
        print "\t".join(poslist)
    quit(1)
      
def Main():
    loadSignalPattern()
    return(1)
                
Main()