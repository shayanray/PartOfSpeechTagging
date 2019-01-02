# Task

Part of Speech Tagging : Perform Feature Engineering, Implement Viterbi decoding, Compare MEMM(Max Entropy Markov Model) to CRF(Conditional Random Fields).

## Config and Results

LR (Token Accuracy: 86.896%) performed a little better than CRF (Token Accuracy: 86.234%) for this
particular feature set however on the whole CRF outperformed LR/MEMM on most occasions and
seems more consistent than LR.
In fact, the sentence accuracy is better for CRF (16.964%) than LR (15.71%) for this best feature-set.


The best selected feature set  was: (both for LR and CRF)

SENTENCE Begin and End, Word case, alphanumeric,
numeric, digit, previous and next word  PLUS (+)

1 WORD_LEMMA
2 HAS_FIRST_LETTER_CAPS
3 HAS_CAPS_IN_BETWEEN
4 HAS_URL
5 HAS_PUNCT
6 HAS_POSITIVE_EMOTICONS
7 HAS_NEGATIVE_EMOTICONS
8 1 CHAR FROM END
9 2 CHARS FROM END
10 3 CHARS FROM END
11 4 CHARS FROM END


