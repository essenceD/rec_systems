import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:

    def __init__(self,
                 weighting: str = 'bm25',
                 n_factors: int = 100,
                 regularization: float = 0.001,
                 iterations: int = 15,
                 num_threads: int = 4):

        self.weighting = weighting
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.num_threads = num_threads

        self.data = None
        self.user_item_matrix = None
        self.id_to_itemid = None
        self.id_to_userid = None
        self.itemid_to_id = None
        self.userid_to_id = None
        self.model = None

    def prepare_matrix(self,
                       index: str = 'user_id',
                       columns: str = 'item_id',
                       values: str = 'quantity'):

        user_item_matrix = pd.pivot_table(self.data,
                                          index=index,
                                          columns=columns,
                                          values=values,
                                          aggfunc='count',
                                          fill_value=0)

        self.user_item_matrix = user_item_matrix.astype(float)

        if self.weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
        if self.weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(user_item_matrix.T).T

    def prepare_dicts(self):

        userids = self.user_item_matrix.index.values
        itemids = self.user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))

    @staticmethod
    def fit_own_recommender(user_item_matrix):

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T)

        return own_recommender

    def fit(self, data):

        self.data = data
        self.prepare_matrix()
        self.prepare_dicts()

        self.model = AlternatingLeastSquares(factors=self.n_factors,
                                             regularization=self.regularization,
                                             iterations=self.iterations,
                                             num_threads=self.num_threads)
        self.model.fit(csr_matrix(self.user_item_matrix).T)

        return self.model

    def get_similar_items_recommendation(self, item, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_rec_list = []
        recs = self.model.similar_items(self.itemid_to_id[item], N=N + 1)
        for i in range(len(recs)):
            top_rec_list.append = self.id_to_itemid[recs[i][0]]
        del top_rec_list[0]
        return top_rec_list


    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        top_rec_list = []
        recs = self.model.similar_users(self.userid_to_id[user], N + 1)[1:]
        similar_users_list = [self.id_to_userid[j] for j in [i[0] for i in recs]]

        for i in similar_users_list:
            recs = self.model.recommend(userid=self.userid_to_id[i],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        # на вход user-item matrix
                                        N=1,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)[0]
            item_id = self.id_to_itemid[recs[0]]
            top_rec_list.append(item_id)

        return top_rec_list
    