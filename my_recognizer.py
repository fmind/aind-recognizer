import logging
import math
import warnings
from functools import partial

from asl_data import SinglesData


def safe_score(model, parameters):
    """Compute the score of a model or return None."""
    try:
        return model.score(*parameters)
    except Exception as e:
        # logging.warn("({}) {}".format(e.__class__.__name__, str(e)))

        return None


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def evaluate(models, params):
        """Return the log-likelihood for params given a list of models."""
        return {word: safe_score(model, params) or -math.inf for word, model in models.items()}

    def predict(evaluation):
        """Predict the best guess word given an evaluations dictionary."""
        # key=evaluation.get: return the dict val
        return max(evaluation, key=evaluation.get)

    # convert the test set into a list of parameters
    parameters = test_set.get_all_Xlengths().values()
    evaluate_given_models = partial(evaluate, models)

    # compute evaluation: log-likelihood / model (word) / parameters
    evaluations = list(map(evaluate_given_models, parameters))
    # compute prediction: max(evaluation.value) / parameters
    predictions = list(map(predict, evaluations))

    return evaluations, predictions
