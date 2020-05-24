######################################################
# function to evaluate success of keyword extraction #
######################################################
# Returns precision, recall, and f1 score for proposed keywords against ground truth
######################################################
def evaluate_keywords(proposed,groundtruth,partMatchesCounted=False):
    proposed_set = set(proposed)
    true_set = set(groundtruth)
  
    true_positives = len(proposed_set.intersection(true_set))
    
    if len(proposed_set)==0:
        precision = 0
    else:
    # note denominator reflects total number of words
    # not total number of unique words
        precision = true_positives/float(len(proposed)) 
    
    if (partMatchesCounted == True):
        lst = list(true_set.difference(proposed_set.intersection(true_set)))
        delta = 0
        for p in proposed:
            for el in p.split(' '):
                d = 0
                for l in lst:
                    if (el in l.split(' ')):
                        if (1/(len(l.split(' '))*len(p.split(' ')))) > d: #(1/len(l.split(' '))) > d:
                            d = 1/(len(l.split(' '))*len(p.split(' ')))
                delta += d
        precision += delta
    
    if (partMatchesCounted == False):
        if len(true_set)==0:
            recall = 0
        else:
            recall = true_positives/float(len(true_set))
    else:
        word_list = [x.split(' ') for x in groundtruth]
        groundtruth_words_set = set([j for i in word_list for j in i])

        word_list = [x.split(' ') for x in proposed]
        proposed_words_set = set([j for i in word_list for j in i])

        true_positives_mod = len(proposed_words_set.intersection(groundtruth_words_set))
        
        if len(groundtruth_words_set)==0:
            recall = 0
        else:
            recall = true_positives_mod/float(len(groundtruth_words_set))

    if precision + recall > 0:
        f1 = 2*precision*recall/float(precision + recall)
    else:
        f1 = 0

    return (precision, recall, f1)