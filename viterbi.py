import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """

    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    # initialize    
    best_sequence = []
    emission_scores_transpose = np.transpose(emission_scores) ## L X N mat
    POS_markers = np.zeros_like(emission_scores_transpose, dtype=int) ## L X N mat  - keep track of POS tags in previous visits
    current_scores = np.zeros_like(emission_scores_transpose) ## L X N mat - keep track of cumulative scores

    #1st column of DP table - initialize with start_scores
    for eachLabel in xrange(L):
        current_scores[eachLabel][0] =  emission_scores_transpose[eachLabel][0] + start_scores[eachLabel]

    # starting from 2nd col onwards
    for currentToken in xrange(1,N): # for each token in the sentence
        for currentLabel in xrange(L): # for each POS tag/label
            current_coln_score = np.zeros(L) ## vector (LX1)
            
            # calculate current column score
            for currentColnLabel in xrange(L):
                current_coln_score[currentColnLabel] = current_scores[currentColnLabel][currentToken-1] + trans_scores[currentColnLabel][currentLabel] + emission_scores_transpose[currentLabel][currentToken]


            #print("current_coln_score >> ", current_coln_score)
            #print("current_coln_score ARGMAX >> ", (current_coln_score).argmax())
            current_scores[currentLabel][currentToken] = np.max(current_coln_score) # find max score in that column #+ emission_scores_transpose[j][i]
            POS_markers[currentLabel][currentToken] = np.argmax(current_coln_score) # find position of max score 


    # add the end scores in the last column of current_scores table
    for eachLabel in xrange(L):
        current_scores[eachLabel][N - 1] += end_scores[eachLabel]

    #print("current_scores >> ",current_scores)
    #print("current_scores[:,N-1] >> ",(current_scores[:,N-1]))

    # it will be the last column having the max value (since its addition of non-zero values)
    best_seq_score = np.max(current_scores[:,N-1])
    pos_best_seq_score = np.argmax(current_scores[:,N-1])

    #print("pos_best_seq_score >> ",pos_best_seq_score)
    #print("POS_markers >> ", POS_markers)
    #print("shape POS_markers>> ", POS_markers.shape)

    best_sequence.append(pos_best_seq_score)

    # go back per token to capture the POS markers with max score
    # go backwards to get the sequence of tag positions from front
    for eachTokenReverse in range(N-1,0,-1):
        pos_best_seq_score=POS_markers[pos_best_seq_score][eachTokenReverse]
        best_sequence.append(pos_best_seq_score)

    
    #print("*********************** ")
    #print("best_seq_score >> ",best_seq_score)
    #print("best_sequence.reverse() >> ",best_sequence[::-1])

    # Finally, return the best viterbi score and the POS tags from start to end (left to right) [reverse it from current state]
    return (best_seq_score, best_sequence[::-1]) 
    