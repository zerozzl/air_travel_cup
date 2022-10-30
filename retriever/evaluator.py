import numpy as np


def evaluate(answers, retrieve_ids, retrieve_scores=None, accept_retrieve_size=0):
    if retrieve_scores is None:
        retrieve_scores = list(range(len(retrieve_ids), 0, -1))

    scores_sort = np.argsort(retrieve_scores)
    scores_sort = scores_sort[::-1]
    if accept_retrieve_size > 0:
        scores_sort = scores_sort[0:accept_retrieve_size]

    sorted_ids = []
    for idx in scores_sort:
        sorted_ids.append(retrieve_ids[idx])

    result = 0
    if len(answers) == 0:
        result = 1
    else:
        for record in sorted_ids:
            for answer in answers:
                if record['content-key'] == answer['content-key'] and record['detail'] == answer['detail']:
                    result = 1
                    break
    return result
