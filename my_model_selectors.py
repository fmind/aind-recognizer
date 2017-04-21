import logging
import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold

from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self,
                 all_word_sequences: dict,
                 all_word_Xlengths: dict,
                 this_word: str,
                 n_constant=3,
                 min_n_components=2,
                 max_n_components=10,
                 random_state=14,
                 verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(
                n_components=num_states,
                covariance_type="diag",
                n_iter=1000,
                random_state=self.random_state,
                verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def safe_score(self, model, parameters):
        """Compute the score of a model or return None."""
        try:
            return model.score(*parameters)
        except Exception as e:
            logging.warn("({}) {}".format(e.__class__.__name__, str(e)))

            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    See this link to find how to calculate the number of parameters (p):
    https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/11
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        def BIC(model):
            """Return the BIC score of a model."""
            # N: number of points
            # M: number of features
            N, M = self.X.shape
            # k: number of components
            k = model.n_components
            # p: number of parameters
            p = k**2 + 2 * M * k - 1
            # log-likelihood of the fitted model
            logL = self.safe_score(model, (self.X, self.lengths))

            return -2 * logL + p * np.log(N) if logL is not None else math.inf

        # create base models from parameters
        models = map(self.base_model, range(self.min_n_components, self.max_n_components + 1))

        # select the model with the lowest BIC score
        return min(models, key=BIC)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)sum(log(p(x(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        def scores(model, params):
            """Generate a list of score for a model."""
            for ps in params:
                yield self.safe_score(model, ps)

        def DIC(model):
            """Return the DIC score of a model."""
            # log-likelihood of the fitted model = log(P(X(i)))
            logL = self.safe_score(model, (self.X, self.lengths))
            # log-likelihood of the other models = log(P(X(all but i)))
            params = [ps for w, ps in self.hwords.items() if w != self.this_word]
            others = [s for s in scores(model, params) if s is not None]
            M = len(others)

            return logL - 1 / (M - 1) * sum(others) if logL is not None else -math.inf

        # create base models from parameters
        models = map(self.base_model, range(self.min_n_components, self.max_n_components + 1))

        # select the model with the highest DIC score
        return max(models, key=DIC)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        def scores(model, sequences, split_method):
            """Generate a list of score for a model."""
            for train_idx, test_idx in split_method(sequences):
                train_X, train_lengths = combine_sequences(train_idx, sequences)
                test_X, test_lengths = combine_sequences(test_idx, sequences)
                model.fit(train_X, train_lengths)

                yield self.safe_score(model, (test_X, test_lengths))

        def AVG(model, n_splits=3):
            """Return the average log-likelihood on cross-validation folds"""
            if len(self.sequences) < n_splits:
                return -math.inf  # not enough samples

            # create the split method
            split = KFold(n_splits=n_splits).split

            return statistics.mean(s for s in scores(model, self.sequences, split) if s is not None)

        # create base models from parameters
        models = map(self.base_model, range(self.min_n_components, self.max_n_components + 1))

        # select the model with the highest AVG score
        return max(models, key=AVG)
