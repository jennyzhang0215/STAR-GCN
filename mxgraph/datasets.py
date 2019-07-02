import io
from mxnet.gluon.utils import check_sha1
from mxgraph.graph import CSRMat, HeterGraph
from zipfile import ZipFile
import warnings
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import gluonnlp as nlp
try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import

DATASET_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'datasets')
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

_word_embedding = nlp.embedding.GloVe('glove.840B.300d')
_tokenizer = nlp.data.transforms.SpacyTokenizer()

MOVIELENS = ['ml-100k', 'ml-1m', 'ml-10m']
OTHER_NAMES = ["douban", "flixster"]


class LoadData(object):
    def __init__(self, name, use_inductive=False, test_ratio=0.1, val_ratio=0.1,
                 inductive_key="item", inductive_node_frac=10, inductive_edge_frac=90,
                 force_download=False, seed=123):
        """

        Parameters
        ----------
        name : str the dataset name
        use_inductive : bool

        test_ratio : decimal (for transductive) if use_input_test_set=True, then this value has no usage
        val_ratio : decimal (for transductive and inductive)

        inductive_key : str (for inductive)
        inductive_node_frac : int (for inductive)
        inductive_edge_frac : int (for inductive)

        force_download : bool
        seed : int
        """
        self._rng = np.random.RandomState(seed=seed)

        self._name = name
        self._root = os.path.join(DATASET_PATH)
        self._raw_data_urls = {'ml-100k': ['ml-100k.zip',
                                           'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                                           ''],
                               'ml-1m': ['ml-1m.zip',
                                         'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                                         ''],
                               'ml-10m': ['ml-10m.zip',
                                          'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
                                          '']}

        assert name in self._raw_data_urls

        self._get_data(force_download)
        print("Starting processing {} ...".format(self._name))
        if self._name== "ml-100k" or self._name == "ml-1m":
            self._data_path = os.path.join(DATASET_PATH, self._name)
        elif self._name == 'ml-10m':
            self._data_path = os.path.join(DATASET_PATH, "ml-10M100K")
        self._load_raw_user_info()
        self._load_raw_movie_info()

        if self._name == 'ml-100k':
            all_train_rating_info = self._load_raw_rates(os.path.join(self._data_path, 'u1.base'), '\t')
            test_rating_info = self._load_raw_rates(os.path.join(self._data_path, 'u1.test'), '\t')
            all_rating_info = pd.concat([all_train_rating_info, test_rating_info])
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            all_rating_info = self._load_raw_rates(os.path.join(self._data_path, 'ratings.dat'), '::')
        else:
            raise NotImplementedError

        self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                                 cmp_col_name="id",
                                                 reserved_ids_set=set(all_rating_info["user_id"].values),
                                                 label="user")
        self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                                  cmp_col_name="id",
                                                  reserved_ids_set=set(all_rating_info["movie_id"].values),
                                                  label="movie")


        ### Generate features
        self.user_features = self._process_user_fea()
        self.item_features = self._process_movie_fea()
        # Map user/movie to the global id
        print("  -----------------")
        print("Generating user id map and movie id map ...")
        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        print("User features: shape ({},{})".format(self.user_features.shape[0], self.user_features.shape[1]))
        print("Item features: shape ({},{})".format(self.item_features.shape[0], self.item_features.shape[1]))

        all_ratings_csr = sp.coo_matrix((
            all_rating_info['rating'].values.astype(np.float32),
            (np.array([self.global_user_id_map[ele] for ele in all_rating_info['user_id']], dtype=np.int32),
             np.array([self.global_movie_id_map[ele] for ele in all_rating_info['movie_id']], dtype=np.int32))),
            shape=(self.num_user, self.num_item), dtype=np.float32).tocsr()
        all_ratings_CSRMat = CSRMat.from_spy(all_ratings_csr)
        self.uniq_ratings = np.unique(all_ratings_CSRMat.values)
        all_ratings_CSRMat.multi_link = self.uniq_ratings

        self._graph = HeterGraph(features={self.name_user: self.user_features,
                                           self.name_item: self.item_features},
                                 csr_mat_dict={(self.name_user, self.name_item): all_ratings_CSRMat},
                                 multi_link={(self.name_user, self.name_item): self.uniq_ratings})


        ### for test and valid set
        self._use_inductive = use_inductive

        if not use_inductive:
            if self._name == 'ml-1m' or self._name == 'ml-10m':
                num_test = int(np.ceil(all_rating_info.shape[0] * test_ratio))
                shuffled_idx = np.random.permutation(all_rating_info.shape[0])
                test_rating_info = all_rating_info.iloc[shuffled_idx[: num_test]]
                all_train_rating_info = all_rating_info.iloc[shuffled_idx[num_test:]]
            num_valid = int(np.ceil(all_train_rating_info.shape[0] * val_ratio))
            shuffled_idx = np.random.permutation(all_train_rating_info.shape[0])
            valid_rating_info = all_train_rating_info.iloc[shuffled_idx[: num_valid]]

            test_node_pairs = np.stack([
                np.array([self.global_user_id_map[ele] for ele in test_rating_info['user_id']], dtype=np.int32),
                np.array([self.global_movie_id_map[ele] for ele in test_rating_info['movie_id']], dtype=np.int32)])
            test_values = test_rating_info['rating'].values.astype(np.float32)
            self._test_data = (test_node_pairs, test_values)
            valid_node_pairs = np.stack([
                np.array([self.global_user_id_map[ele] for ele in valid_rating_info['user_id']], dtype=np.int32),
                np.array([self.global_movie_id_map[ele] for ele in valid_rating_info['movie_id']], dtype=np.int32)])
            valid_values = test_rating_info['rating'].values.astype(np.float32)
            self._valid_data = (valid_node_pairs, valid_values)
        else:
            self._inductive_key = inductive_key
            self._inductive_node_frac = inductive_node_frac
            self._inductive_edge_frac = inductive_edge_frac
            if inductive_key == "item":
                self._inductive_key = self.name_item
            elif inductive_key == "user":
                self._inductive_key = self.name_user
            else:
                raise NotImplementedError
            all_node_ids = self.graph.node_ids[self._inductive_key]
            train_val_ids, self._inductive_test_ids, self._test_data = \
                self._gen_inductive_data(all_node_ids)
            self._inductive_train_ids, self._inductive_valid_ids, self._valid_data = \
                self._gen_inductive_data(train_val_ids)


    def _gen_inductive_data(self, node_ids):
        total_num_nodes = node_ids.shape[0]
        shuffled_idx = np.random.permutation(total_num_nodes).astype(np.int32)
        test_num = int(np.ceil(total_num_nodes / 100.0 * self._inductive_node_frac))
        shuffled_nodes = node_ids[shuffled_idx]
        count_test_id = 0
        test_ids = []
        train_ids = []
        test_rating_pairs_l = []
        ### split the edges for each node
        for idx, ele_id in enumerate(shuffled_nodes):
            if self._inductive_key == self.name_user:
                ##(2, #edge)
                node_pairs = self.graph[self.name_user, self.name_item].submat_by_id(
                    row_ids=np.array(ele_id, dtype=np.int32)).node_pair_ids
            elif self._inductive_key == self.name_item:
                node_pairs = self.graph[self.name_user, self.name_item].submat_by_id(
                    col_ids=np.array(ele_id, dtype=np.int32)).node_pair_ids
            else:
                raise NotImplementedError
            num_pair = node_pairs.shape[1]
            assert num_pair != 0
            if num_pair < 10 or num_pair == 10:
                train_ids.append(ele_id)
            else:
                test_ids.append(ele_id)
                count_test_id += 1
                shuffled_pair_idx = np.random.permutation(num_pair).astype(np.int32)
                chosen_num = int(np.floor(num_pair / 100.0 * self._inductive_edge_frac))
                test_node_pair_idx = shuffled_pair_idx[ : chosen_num]
                test_rating_pairs_l.append(node_pairs[:, test_node_pair_idx])
            if count_test_id == test_num:
                break
        test_ids = np.array(test_ids, dtype=np.int32)
        train_ids = np.concatenate((np.array(train_ids, dtype=np.int32), shuffled_idx[idx+1: ]))
        assert shuffled_idx.size == train_ids.size + test_ids.size
        test_rating_pairs = np.hstack(test_rating_pairs_l).astype(np.int32)
        test_data = (test_rating_pairs,
                     self._graph.fetch_edges_by_id(src_key=self.name_user,
                                                   dst_key=self.name_item,
                                                   node_pair_ids=test_rating_pairs))
        return train_ids, test_ids, test_data

    @property
    def name_user(self):
        return 'user'
    @property
    def name_item(self):
        if self._name in MOVIELENS:
            return 'movie'
        elif self._name in OTHER_NAMES:
            return "item"
        else:
            raise NotImplementedError
    @property
    def num_links(self):
        return self.uniq_ratings
    @property
    def num_user(self):
        return len(self.global_user_id_map)
    @property
    def num_item(self):
        return len (self.global_movie_id_map)
    @property
    def inductive_test_ids(self):
        return self._inductive_test_ids
    @property
    def inductive_train_ids(self):
        return self._inductive_train_ids
    @property
    def inductive_valid_ids(self):
        return self._inductive_valid_ids


    @property
    def graph(self):
        """

        Returns
        -------
        ret : HeterGraph
            The inner heterogenous graph
        """
        return self._graph

    @property
    def valid_data(self):
        """

        Returns
        -------
        node_pair_ids : np.ndarray
            Shape (2, TOTAL_NUM)
            First row --> user_id
            Second row --> item_id
        ratings : np.ndarray
            Shape (TOTAL_NUM,)
        """
        return self._valid_data
    @property
    def test_data(self):
        """

        Returns
        -------
        node_pair_ids : np.ndarray
            Shape (2, TOTAL_NUM)
            First row --> user_id
            Second row --> item_id
        ratings : np.ndarray
            Shape (TOTAL_NUM,)
        """
        return self._test_data


    def _get_data(self, force_download=False):
        if not os.path.exists(self._root):
            os.makedirs(self._root)
        data_name, url, data_hash = self._raw_data_urls[self._name]
        path = os.path.join(self._root, data_name)
        if not os.path.exists(path) or force_download or (data_hash and not check_sha1(path, data_hash)):
            print("\n\n=====================> Download dataset ...")
            self.download(url, path=path, sha1_hash=data_hash)
            print("\n\n=====================> Unzip the file ...")
            with ZipFile(path, 'r') as zf:
                zf.extractall(path=self._root)

    def download(self, url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True, proxy_dict=None):
        """Download an given URL

        Parameters
        ----------
        url : str
            URL to download
        path : str, optional
            Destination path to store downloaded file. By default stores to the
            current directory with same name as in url.
        overwrite : bool, optional
            Whether to overwrite destination file if already exists.
        sha1_hash : str, optional
            Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
            but doesn't match.
        retries : integer, default 5
            The number of times to attempt the download in case of failure or non 200 return codes
        verify_ssl : bool, default True
            Verify SSL certificates.

        Returns
        -------
        str
            The file path of the downloaded file.
        """
        if path is None:
            fname = url.split('/')[-1]
            # Empty filenames are invalid
            assert fname, 'Can\'t construct file-name from this URL. ' \
                          'Please set the `path` option manually.'
        else:
            path = os.path.expanduser(path)
            if os.path.isdir(path):
                fname = os.path.join(path, url.split('/')[-1])
            else:
                fname = path
        assert retries >= 0, "Number of retries should be at least 0"

        if not verify_ssl:
            warnings.warn(
                'Unverified HTTPS request is being made (verify_ssl=False). '
                'Adding certificate verification is strongly advised.')
        if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
            dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            while retries + 1 > 0:
                # Disable pyling too broad Exception
                # pylint: disable=W0703
                try:
                    print('Downloading %s from %s...' % (fname, url))
                    try:
                        r = requests.get(url, stream=True, verify=verify_ssl)
                    except:
                        r = requests.get(url, stream=True, verify=verify_ssl, proxies=proxy_dict)
                    if r.status_code != 200:
                        raise RuntimeError("Failed downloading url %s" % url)
                    with open(fname, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                    if sha1_hash and not check_sha1(fname, sha1_hash):
                        raise UserWarning('File {} is downloaded but the content hash does not match.' \
                                          ' The repo may be outdated or download may be incomplete. ' \
                                          'If the "repo_url" is overridden, consider switching to ' \
                                          'the default repo.'.format(fname))
                    break
                except Exception as e:
                    retries -= 1
                    if retries <= 0:
                        raise e
                    else:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))
        return fname




    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        #print("  -----------------")
        #print("{}: {}(reserved) v.s. {}(from info)".format(label, len(reserved_ids_set),
        #                                                     len(set(orign_info[cmp_col_name].values))))
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            #print("\torign_info: ({}, {})".format(orign_info.shape[0], orign_info.shape[1]))
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            #print("\tAfter dropping, data shape: ({}, {})".format(data_info.shape[0], data_info.shape[1]))
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

    def _load_raw_rates(self, file_path, sep):
        """In MovieLens, the rates have the following format

        ml-100k
        user id \t movie id \t rating \t timestamp

        ml-1m/10m
        UserID::MovieID::Rating::Timestamp

        timestamp is unix timestamp and can be converted by pd.to_datetime(X, unit='s')

        Parameters
        ----------
        file_path : str

        Returns
        -------
        rating_info : pd.DataFrame
        """
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python')
        return rating_info

    def _load_raw_user_info(self):
        """In MovieLens, the user attributes file have the following formats:

        ml-100k:
        user id | age | gender | occupation | zip code

        ml-1m:
        UserID::Gender::Age::Occupation::Zip-code

        For ml-10m, there is no user information. We read the user id from the rating file.

        Parameters
        ----------
        name : str

        Returns
        -------
        user_info : pd.DataFrame
        """
        if self._name == 'ml-100k':
            self.user_info = pd.read_csv(os.path.join(self._data_path, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        elif self._name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(self._data_path, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        elif self._name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(self._data_path, 'ratings.dat'), sep='::', header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                dtype={'user_id': np.int32, 'movie_id': np.int32, 'ratings': np.float32,
                       'timestamp': np.int64}, engine='python')
            self.user_info = pd.DataFrame(np.unique(rating_info['user_id'].values.astype(np.int32)),
                                     columns=['id'])
        else:
            raise NotImplementedError

    def _process_user_fea(self):
        """

        Parameters
        ----------
        user_info : pd.DataFrame
        name : str
        For ml-100k and ml-1m, the column name is ['id', 'gender', 'age', 'occupation', 'zip_code'].
            We take the age, gender, and the one-hot encoding of the occupation as the user features.
        For ml-10m, there is no user feature and we set the feature to be a single zero.

        Returns
        -------
        user_features : np.ndarray

        """
        if self._name == 'ml-100k' or self._name == 'ml-1m':
            ages = self.user_info['age'].values.astype(np.float32)
            gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
            all_occupations = set(self.user_info['occupation'])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)),
                                          dtype=np.float32)
            occupation_one_hot[np.arange(self.user_info.shape[0]),
                               np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
            user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                            gender.reshape((self.user_info.shape[0], 1)),
                                            occupation_one_hot], axis=1)
        elif self._name == 'ml-10m':
            user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)
        else:
            raise NotImplementedError
        return user_features

    def _load_raw_movie_info(self):
        """In MovieLens, the movie attributes may have the following formats:

        In ml_100k:

        movie id | movie title | release date | video release date | IMDb URL | [genres]

        In ml_1m, ml_10m:

        MovieID::Title (Release Year)::Genres

        Also, Genres are separated by |, e.g., Adventure|Animation|Children|Comedy|Fantasy

        Parameters
        ----------
        name : str

        Returns
        -------
        movie_info : pd.DataFrame
            For ml-100k, the column name is ['id', 'title', 'release_date', 'video_release_date', 'url'] + [GENRES (19)]]
            For ml-1m and ml-10m, the column name is ['id', 'title'] + [GENRES (18/20)]]
        """
        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        if self._name == 'ml-100k':
            file_path = os.path.join(self._data_path, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            file_path = os.path.join(self._data_path, 'movies.dat')
            movie_info = pd.read_csv(file_path, sep='::', header=None,
                                     names=['id', 'title', 'genres'], engine='python')
            genre_map = {ele: i for i, ele in enumerate(GENRES)}
            genre_map['Children\'s'] = genre_map['Children']
            genre_map['Childrens'] = genre_map['Children']
            movie_genres = np.zeros(shape=(movie_info.shape[0], len(GENRES)), dtype=np.float32)
            for i, genres in enumerate(movie_info['genres']):
                for ele in genres.split('|'):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        print('genres not found, filled with unknown: {}'.format(genres))
                        movie_genres[i, genre_map['unknown']] = 1.0
            for idx, genre_name in enumerate(GENRES):
                assert idx == genre_map[genre_name]
                movie_info[genre_name] = movie_genres[:, idx]
            self.movie_info = movie_info.drop(columns=["genres"])
        else:
            raise NotImplementedError

    def _process_movie_fea(self):
        """

        Parameters
        ----------
        movie_info : pd.DataFrame
        name :  str

        Returns
        -------
        movie_features : np.ndarray
            Generate movie features by concatenating embedding and the year

        """
        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)
        p = re.compile(r'(.+)\s*\((\d+)\)')
        for i, title in enumerate(self.movie_info['title']):
            match_res = p.match(title)
            if match_res is None:
                print('{} cannot be matched, index={}, name={}'.format(title, i, self._name))
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            title_embedding[i, :] =_word_embedding[_tokenizer(title_context)].asnumpy().mean(axis=0)
            release_years[i] = float(year)
        movie_features = np.concatenate((title_embedding,
                                         (release_years - 1950.0) / 100.0,
                                         self.movie_info[GENRES]),
                                        axis=1)
        return movie_features



    def __repr__(self):
        stream = io.StringIO()
        print('Dataset Name={}'.format(self._name), file=stream)
        print(self.graph, file=stream)
        print('#Val/Test edges: {}/{}'.format(self.valid_data[1].size, self.test_data[1].size),
              file=stream)
        if self._use_inductive:
            print('Inductive splits:', file=stream)
            print('\tNode ratio={}%, # Train/Valid/Test nodes: {}/{}/{}'
                  .format(self._inductive_node_frac,
                          self.inductive_train_ids.size, self.inductive_valid_ids.size, self.inductive_test_ids.size),
                  file=stream)
            print('\tEdge ratio={}%, # Train/Valid/Test pairs: {}/{}/{}'
                  .format(self._inductive_edge_frac,
                          self._graph[self.name_user, self.name_item].values.size -
                          self.valid_data[1].size - self.test_data[1].size,
                          self.valid_data[1].size, self.test_data[1].size),
                  file=stream)
        print('------------------------------------------------------------------------------',
              file=stream)
        return stream.getvalue()


if __name__ == '__main__':
    #LoadData('ml-100k', use_inductive=False, test_ratio=0.2, val_ratio=0.1, seed=100)
    #LoadData('ml-1m', use_inductive=False, test_ratio=0.1, val_ratio=0.1, seed=100)
    LoadData('ml-10m', use_inductive=False, test_ratio=0.1, val_ratio=0.1, seed=100)
