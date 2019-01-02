#!/bin/python
import nltk.stem
import string
import nltk.corpus

positive_emoticons = [":-)",":)",":-]",":]",":-3",":3",":->",":>","8-)","8)",":-}",":}",":o)",":c)",":^)","=]","=)",":-D",":D","8-D","8D","x-D","xD","X-D","XD","=D","=3","B^D",":-))",";-)",";)","*-)","*)",";-]",";]",";^)",";D",":-P",":P","X-P","x-p",":-p",":p",":-?",":?",":-?",":?",":-b",":b","=p",">:P",":*",":-*","^.^","^_^","^-^","xd"]
negative_emoticons = [":-(",":(",":-c",":c",":-<",":<",":-[",":[",":-||",">:[",":{",":@",">:(",":-/",":/",">:\\",">:/","=/",":L","=L",":S",":-|",":|",":-X",":X","-.-","-,-","=\\",":\\"] 


def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    #pass

    #print("DEBUG ...................................... ")
    #for aSentence in train_sents:
    #    print(aSentence)
    #print("END DEBUG ...................................... ")

    # remove extra spaces, tabs, newlines, carriage return etc.
    train_sents = [[aWord.strip(' \t\n\r')  for aWord in aSentence] for aSentence in train_sents]
    #print("After preprocess_corpus >> ",train_sents)


def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    word = word.strip(' \t\n\r')  #### added - cleanedup unnecessary tabs, spaces, carriage returns
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    ##------ Start of adding Custom Features --------------------------------
    ## 1. Lemma of the word

    word_lwr = word.lower()
    wnl_lemmatzr = nltk.stem.WordNetLemmatizer() 
    try:
        ftrs.append(wnl_lemmatzr.lemmatize(word_lwr))
    except Exception:    
        nltk.download('wordnet')
        ftrs.append(wnl_lemmatzr.lemmatize(word_lwr))

    #2. first letter caps?
    if word[0] ==word[0].upper() and word.upper() != word:
        ftrs.append("HAS_FIRST_LETTER_CAPS")

    #3. intermediate CAPS?
    if word[1:] != word_lwr[1:]:
        ftrs.append("HAS_CAPS_IN_BETWEEN")

    #4. check URL
    check1 = "http://"
    check2 = "https://"
    len_check = len(check1)
    if (check1 in word_lwr or check2 in word_lwr ) and len(word_lwr) > len_check:
        ftrs.append("HAS_URL")

    
    #5. check punctuations includes    !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    check = set(string.punctuation)
    if any((c in check) for c in word):
        ftrs.append("HAS_PUNCT")

    #6. Emotions
    
    words = word.split('\\s+')
    if words[0] in positive_emoticons:
        ftrs.append("HAS_POSITIVE_EMOTICONS")

    if words[0] in negative_emoticons:
        ftrs.append("HAS_NEGATIVE_EMOTICONS")

    '''
    #7. check valid word
    if nltk.corpus.wordnet.synsets(word):
        ftrs.append("IS_VALID_WORD")
    '''
    
    #8. get the last few characters of the word - this will capture plurals, verbs, word endings
    ##### last 2 chars gives the BEST results for LR/MEMM on DEV dataset but not much difference from last 4 chars.
    ##### last 4 chars gives the BEST results for CRF on DEV dataset
    ##### submitting 4 chars as final CHOSEN feature set - along with already existing features as both models show 86.xx % token accuracy with this feature set.
    
    if len(word) > 0:
        ftrs.append(word_lwr[-1:])

    if len(word) > 1:
        ftrs.append(word_lwr[-2:])

    
    if len(word) > 2:
        ftrs.append(word_lwr[-3:])


    if len(word) > 3:
        ftrs.append(word_lwr[-4:])

    '''
    if len(word)>4:
        ftrs.append(word_lwr[-5:])
    '''

    ##------ End of adding Custom Features --------------------------------
    # previous/next word feats
    if add_neighs:
        if i > 0: #1
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1: #2
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    '''
    sents = [
    [ "I", "love   ", "food", ":)" ]
    ]
    '''
    sents = [
    ["@paulwalk", "It", "'s", "the", "view", "from", "where", "I", "'m", "living", "for", "two", "weeks", ".", "Empire", "State", "Building", "=", "ESB", ".", "Pretty", "bad", "storm", "here", "last", "evening", "."],
    ["Small", "Biz", "Tech", "Tour", "2010", "Launches", "Five", "City", "Tour", "MONTCLAIR", "N.J.", "...:", "The", "all", "day", "event", "features", "America's...", "http://tinyurl.com/28hd9f", "#fb"],
    ["@MiSS_SOTO", "I", "think", "that", "'s", "when", "I", "'m", "gonna", "be", "there"],
    ["On", "Thanksgiving", "after", "yo", "done", "eating", "its", "#TimeToGetOut", "unless", "yo", "wanna", "help", "with", "the", "dishes"]
    ]
    
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
