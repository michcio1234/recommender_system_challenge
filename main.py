import cPickle as pickle

import numpy as np
import pandas as pd
from lightfm import LightFM
from scipy.sparse import coo_matrix, identity, lil_matrix

"""
This is a solution to Siraj Raval's challenge on recommendation system.
You can obtain an Amazon dataset from http://jmcauley.ucsd.edu/data/amazon/links.html (ratings only).

Author: michcio1234
"""

RATINGS_CSV_FILE = "ratings_Electronics.csv"  # read data from this file
PICKLED_DATA_FILE = 'data.pck'  # save data in under this path to avoid parsing CSV every time
MINIMUM_INTERACTIONS = 50  # take into account only users with at least this number of interactions (ratings)
THR = 5.0  # take into account rating greater than or equal to this threshold
USERS_TO_SAMPLE_RECOMMENDATIONS_FOR = [3, 25, 450]


def read_data(file_path, min_interactions=0, force=False):

    # load the pickled file if it exists
    if not force:
        try:
            with open(PICKLED_DATA_FILE, 'rb') as f:
                print "Getting the data from the pickled file..."
                data = pickle.load(f)
                return data
        except IOError:
            print "The file data.pck was not found. Generating it..."

    # read the full file
    with open(file_path) as f:
        original_data = pd.read_csv(f, header=None, names=['user', 'item', 'rating', 'timestamp'])

    # select only those users who appear at least min_interactions times
    user_count = original_data['user'].value_counts()
    filtered_users = user_count[user_count >= min_interactions].index
    filtered_original_data = original_data[original_data['user'].isin(filtered_users)]
    # select only those interactions which has a rating >= THR
    filtered_original_data = filtered_original_data[filtered_original_data['rating'] >= THR]
    # create a dict similar to the original one (MovieLens dataset)
    data = dict()
    data['item_labels'] = filtered_original_data['item'].unique()
    data['item_features'] = identity(len(data['item_labels']), format='csr')
    data['item_feature_labels'] = data['item_labels']
    # create sparse interactions matrix
    interactions = lil_matrix((len(filtered_users), len(data['item_labels'])), dtype=np.int32)
    n = 0
    for index, row in filtered_original_data.iterrows():
        interactions[filtered_users.get_loc(row['user']),
                     np.where(data['item_labels'] == row['item'])[0]] = row['rating']
        n += 1
        if n % 100 == 0:  # print progress every 100 entries
            print "{}/{}".format(n, len(filtered_original_data))
    data['interactions'] = interactions.tocoo()

    # save data for future use
    with open(PICKLED_DATA_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def split_data(input_matrix, split_fraction=0.5):
    # type: (coo_matrix) -> (coo_matrix, coo_matrix)
    """
    Split each user's interactions into training and testing set.

    :param input_matrix: sparse interactions matrix
    :param split_fraction: a fraction of each user's interactions to be taken into training set
    :return: a tuple of two matrices: training and testing
    """
    train = input_matrix.tolil()
    for row, data in zip(train.rows, train.data):
        del row[:int(split_fraction * len(row))]
        del data[:int(split_fraction * len(data))]
    test = input_matrix.tolil()
    for row, data in zip(test.rows, test.data):
        del row[int(split_fraction * len(row)):]
        del data[int(split_fraction * len(data)):]
    return train.tocoo(), test.tocoo()


def check_efficiency(model, interactions):
    """Calculates efficiency of a model given interactions matrix.
    Efficiency is a mean score our model predicts for all items which were ranked at least THR by a user.
    """
    # number of users and items in testing data
    n_users, n_items = interactions.shape

    efficiency_score = 0.0
    for user_id in range(n_users):

        # scores our model predicts
        scores_predict = model.predict(user_id, np.arange(n_items))
        # true scores of items from testing set
        scores_true = np.asarray(interactions.getrow(user_id).todense())[0]
        # efficiency score as a mean predicted score of items rated higher or equal than threshold
        efficiency_score += np.mean(scores_predict[scores_true >= THR])

    return efficiency_score / n_users


def sample_recommendation(model, data, user_ids):
    # number of users and items in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:

        # items they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # items our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


if __name__ == '__main__':
    # prepare data
    data = read_data(RATINGS_CSV_FILE, min_interactions=MINIMUM_INTERACTIONS)
    data['train'], data['test'] = split_data(data['interactions'])

    # create models for every loss function
    loss_functions = ['warp', 'logistic', 'bpr', 'warp-kos']
    models = {loss_function: LightFM(loss=loss_function) for loss_function in loss_functions}

    # train models
    for loss_function, model in models.iteritems():
        print "Training model {}...".format(loss_function)
        model.fit(data['train'], epochs=30, num_threads=4)
        print "Trained"

    print "Calculating efficiencies..."
    models_efficiency = {}
    for name, model in models.iteritems():
        models_efficiency[name] = check_efficiency(model, data['test'])
    max_efficiency_model_name = max(models_efficiency.iterkeys(), key=lambda key: models_efficiency[key])

    print "Models efficiencies:"
    for name, efficiency in models_efficiency.iteritems():
        print "\t{}:\t{:.2f}{}".format(name, efficiency, " (best)" if name == max_efficiency_model_name else "")

    print "Best model's recommendations:"
    sample_recommendation(models[max_efficiency_model_name], data, USERS_TO_SAMPLE_RECOMMENDATIONS_FOR)
