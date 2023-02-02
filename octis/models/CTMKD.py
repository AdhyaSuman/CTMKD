from sklearn.feature_extraction.text import CountVectorizer

from octis.models.model import AbstractModel
from octis.models.contextualized_topic_models_KD.datasets import dataset
from octis.models.contextualized_topic_models_KD.models import ctmkd
from octis.models.contextualized_topic_models_KD.utils.data_preparation import bert_embeddings_from_list

import os
import pickle as pkl


class CTMKD(AbstractModel):

    def __init__(self, num_topics=10, model_type='prodLDA', activation='softplus',
                 dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False, prior_mean=0.0,
                 prior_variance=None,
                 hidden_sizes=None, num_layers=None, num_neurons=None,
                 use_partitions=True, use_validation=True,
                 num_samples=10,
                 inference_type="zeroshot",
                 pre_processing_type='KD',
                 student_bert_path="",
                 teacher_bert_path="",
                 student_bert_model="distiluse-base-multilingual-cased-v2",
                 teacher_bert_model="paraphrase-distilroberta-base-v2",
                 teacher=None, temp=2.0, alpha=0.4, t_beta=None,
                 use_topic_vector_kd=False, use_mean_logvar_kd=False,
                 use_mean_logvar_recon_kd=True, KD_epochs=10, KD_loss_type='2wd'):
        """
        initialization of CTMKD

        :param num_topics : int, number of topic components, (default 10)
        :param model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param activation : string, 'softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu',
        'selu' (default 'softplus')
        :param num_layers : int, number of layers (default 2)
        :param dropout : float, dropout to use (default 0.2)
        :param learn_priors : bool, make priors a learnable parameter (default True)
        :param batch_size : int, size of batch to use for training (default 64)
        :param lr : float, learning rate to use for training (default 2e-3)
        :param momentum : float, momentum to use for training (default 0.99)
        :param solver : string, optimizer 'adam' or 'sgd' (default 'adam')
        :param num_epochs : int, number of epochs to train for, (default 100)
        :param num_samples: int, number of times theta needs to be sampled (default: 10)
        :param use_partitions: bool, if true the model will be trained on the training set and evaluated on the test
        set (default: true)
        :param reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
        :param inference_type: the type of the CTMKD model. It can be "zeroshot" or "combined" (default zeroshot)
        :param bert_path: path to store the document contextualized representations
        :param bert_model: name of the contextualized model (default: bert-base-nli-mean-tokens).
        see https://www.sbert.net/docs/pretrained_models.html
        """

        super().__init__()

        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['model_type'] = model_type
        self.hyperparameters['activation'] = activation
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['inference_type'] = inference_type
        self.hyperparameters['learn_priors'] = learn_priors
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['lr'] = lr
        self.hyperparameters['num_samples'] = num_samples
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['solver'] = solver
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['reduce_on_plateau'] = reduce_on_plateau
        self.hyperparameters["prior_mean"] = prior_mean
        self.hyperparameters["prior_variance"] = prior_variance

        self.hyperparameters["student_bert_path"] = student_bert_path
        self.hyperparameters["teacher_bert_path"] = teacher_bert_path

        self.hyperparameters["student_bert_model"] = student_bert_model
        self.hyperparameters["teacher_bert_model"] = teacher_bert_model

        self.hyperparameters["teacher"] = teacher
        self.hyperparameters["temp"] = temp
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["t_beta"] = t_beta
        self.hyperparameters["use_topic_vector_kd"] = use_topic_vector_kd
        self.hyperparameters["use_mean_logvar_kd"] = use_mean_logvar_kd
        self.hyperparameters["use_mean_logvar_recon_kd"] = use_mean_logvar_recon_kd
        self.hyperparameters["KD_epochs"] = KD_epochs
        self.hyperparameters["KD_loss_type"] = KD_loss_type

        self.use_partitions = use_partitions
        self.use_validation = use_validation
        self.pre_processing_type = pre_processing_type

        self.hyperparameters["num_neurons"] = num_neurons
        self.hyperparameters["num_layers"] = num_layers
        self.hyperparameters['hidden_sizes'] = CTMKD.set_hiddensize(hidden_sizes, num_neurons, num_layers)

        self.model = None
        self.vocab = None

    @staticmethod
    def set_hiddensize(hidden_sizes=None, num_neurons=None, num_layers=None):
        '''
        if [num_neurons=100, num_layers=2 and hidden_sizes=None], then hidden_sizes=(100,100)
        if [(num_neurons and num_layers)=None and hidden_sizes=(50,100)], then hidden_sizes=(50,100)
        if [(num_neurons and num_layers)=None and hidden_sizes=None], then Error
        if [num_neurons=100, num_layers=2 and hidden_sizes=(50,100)], then Missmatch Error
        '''
        H_SIZES = None
        if (num_layers and num_neurons) != None:
            if hidden_sizes == None:
                H_SIZES = tuple([num_neurons for _ in range(num_layers)])
            else:
                if hidden_sizes == tuple([num_neurons for _ in range(num_layers)]):
                    H_SIZES = hidden_sizes
                else:
                    raise Exception('Missmatch, Should be::: hidden_sizes == tuple([num_neurons for _ in range(num_layers)])')

        else:
            if hidden_sizes == None:
                H_SIZES=None
            else:
                H_SIZES = hidden_sizes
        return H_SIZES


    def train_model(self, dataset, hyperparameters=None, top_words=10, save_dir=None):
        """
        trains CTMKD model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)

        """
        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.vocab = dataset.get_vocabulary()

        if self.use_partitions and self.use_validation:
            train, validation, test = dataset.get_partitioned_corpus(use_validation=True)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]
            data_corpus_validation = [' '.join(i) for i in validation]

            x_train, x_test, x_valid, input_size = self.preprocess(
                self.vocab, data_corpus_train, test=data_corpus_test, validation=data_corpus_validation,
                Sbert_train_path=self.hyperparameters['student_bert_path'] + "_train.pkl",
                Sbert_test_path=self.hyperparameters['student_bert_path'] + "_test.pkl",
                Sbert_val_path=self.hyperparameters['student_bert_path'] + "_val.pkl",
                Sbert_model=self.hyperparameters["student_bert_model"],
                Tbert_train_path=self.hyperparameters['teacher_bert_path'] + "_train.pkl",
                Tbert_test_path=self.hyperparameters['teacher_bert_path'] + "_test.pkl",
                Tbert_val_path=self.hyperparameters['teacher_bert_path'] + "_val.pkl",
                Tbert_model=self.hyperparameters["teacher_bert_model"],
                type=self.pre_processing_type)

            self.model = ctmkd.CTMKD(input_size=input_size, bert_input_size=x_train.X_bert.shape[1],
                                 model_type=self.hyperparameters['model_type'],
                                 num_topics=self.hyperparameters['num_topics'], dropout=self.hyperparameters['dropout'],
                                 activation=self.hyperparameters['activation'], lr=self.hyperparameters['lr'],
                                 inference_type=self.hyperparameters['inference_type'],
                                 hidden_sizes=self.hyperparameters['hidden_sizes'],
                                 solver=self.hyperparameters['solver'],
                                 momentum=self.hyperparameters['momentum'],
                                 num_epochs=self.hyperparameters['num_epochs'],
                                 learn_priors=self.hyperparameters['learn_priors'],
                                 batch_size=self.hyperparameters['batch_size'],
                                 num_samples=self.hyperparameters['num_samples'],
                                 topic_prior_mean=self.hyperparameters["prior_mean"],
                                 reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                                 topic_prior_variance=self.hyperparameters["prior_variance"],
                                 KD_loss_type=self.hyperparameters["KD_loss_type"])

            self.model.fit(x_train, x_valid, teacher=self.hyperparameters["teacher"], alpha=self.hyperparameters["alpha"],
                           temp=self.hyperparameters["temp"], use_topic_vector_kd=self.hyperparameters["use_topic_vector_kd"],
                           use_mean_logvar_kd=self.hyperparameters["use_mean_logvar_kd"],
                           use_mean_logvar_recon_kd=self.hyperparameters["use_mean_logvar_recon_kd"],
                           KD_epochs=self.hyperparameters["KD_epochs"], save_dir=save_dir, texts=train)

            result = self.inference(x_test)
            return result
        
        elif self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]

            x_train, x_test, input_size = self.preprocess(
                self.vocab, data_corpus_train, test=data_corpus_test, validation=None,
                Sbert_train_path=self.hyperparameters['student_bert_path'] + "_train.pkl",
                Sbert_test_path=self.hyperparameters['student_bert_path'] + "_test.pkl",
                Sbert_val_path=None,
                Sbert_model=self.hyperparameters["student_bert_model"],
                Tbert_train_path=self.hyperparameters['teacher_bert_path'] + "_train.pkl",
                Tbert_test_path=self.hyperparameters['teacher_bert_path'] + "_test.pkl",
                Tbert_val_path=None,
                Tbert_model=self.hyperparameters["teacher_bert_model"],
                type=self.pre_processing_type)

            self.model = ctmkd.CTMKD(input_size=input_size, bert_input_size=x_train.X_bert.shape[1],
                                 model_type=self.hyperparameters['model_type'],
                                 num_topics=self.hyperparameters['num_topics'], dropout=self.hyperparameters['dropout'],
                                 activation=self.hyperparameters['activation'], lr=self.hyperparameters['lr'],
                                 inference_type=self.hyperparameters['inference_type'],
                                 hidden_sizes=self.hyperparameters['hidden_sizes'],
                                 solver=self.hyperparameters['solver'],
                                 momentum=self.hyperparameters['momentum'],
                                 num_epochs=self.hyperparameters['num_epochs'],
                                 learn_priors=self.hyperparameters['learn_priors'],
                                 batch_size=self.hyperparameters['batch_size'],
                                 num_samples=self.hyperparameters['num_samples'],
                                 topic_prior_mean=self.hyperparameters["prior_mean"],
                                 reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                                 topic_prior_variance=self.hyperparameters["prior_variance"],
                                 KD_loss_type=self.hyperparameters["KD_loss_type"])

            self.model.fit(x_train, teacher=self.hyperparameters["teacher"], alpha=self.hyperparameters["alpha"],
                           temp=self.hyperparameters["temp"], use_topic_vector_kd=self.hyperparameters["use_topic_vector_kd"],
                           use_mean_logvar_kd=self.hyperparameters["use_mean_logvar_kd"],
                           use_mean_logvar_recon_kd=self.hyperparameters["use_mean_logvar_recon_kd"],
                           KD_epochs=self.hyperparameters["KD_epochs"], save_dir=save_dir, texts=train)

            result = self.inference(x_test)
            return result

        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            x_train, input_size = self.preprocess(
                self.vocab, train=data_corpus, Sbert_train_path=self.hyperparameters['student_bert_path'] + "_train.pkl",
                Sbert_model=self.hyperparameters["student_bert_model"],
                Tbert_train_path=self.hyperparameters['teacher_bert_path'] + "_train.pkl",
                Tbert_model=self.hyperparameters["teacher_bert_model"],
                type=self.pre_processing_type)

        self.model = ctmkd.CTMKD(input_size=input_size, bert_input_size=x_train.X_bert.shape[1], model_type='prodLDA',
                             num_topics=self.hyperparameters['num_topics'], dropout=self.hyperparameters['dropout'],
                             activation=self.hyperparameters['activation'], lr=self.hyperparameters['lr'],
                             inference_type=self.hyperparameters['inference_type'],
                             hidden_sizes=self.hyperparameters['hidden_sizes'], solver=self.hyperparameters['solver'],
                             momentum=self.hyperparameters['momentum'], num_epochs=self.hyperparameters['num_epochs'],
                             learn_priors=self.hyperparameters['learn_priors'],
                             batch_size=self.hyperparameters['batch_size'],
                             num_samples=self.hyperparameters['num_samples'],
                             topic_prior_mean=self.hyperparameters["prior_mean"],
                             reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                             topic_prior_variance=self.hyperparameters["prior_variance"],
                             KD_loss_type=self.hyperparameters["KD_loss_type"])


        self.model.fit(x_train, None, teacher=self.hyperparameters["teacher"], alpha=self.hyperparameters["alpha"],
                           temp=self.hyperparameters["temp"], use_topic_vector_kd=self.hyperparameters["use_topic_vector_kd"],
                           use_mean_logvar_kd=self.hyperparameters["use_mean_logvar_kd"],
                           use_mean_logvar_recon_kd=self.hyperparameters["use_mean_logvar_recon_kd"],
                           KD_epochs=self.hyperparameters["KD_epochs"], save_dir=save_dir, texts=dataset.get_corpus())
                            
        result = self.model.get_info()
        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
             if k in self.hyperparameters.keys() and k != 'hidden_sizes':
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])
                
        if 'hidden_sizes' in hyperparameters.keys():
            self.hyperparameters['hidden_sizes'] = CTMKD.set_hiddensize(hyperparameters['hidden_sizes'], self.hyperparameters['num_neurons'], self.hyperparameters['num_layers'])

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    @staticmethod
    def preprocess(vocab, train, Sbert_model, Tbert_model, test=None, validation=None,
                   Sbert_train_path=None, Sbert_test_path=None, Sbert_val_path=None,
                   Tbert_train_path=None, Tbert_test_path=None, Tbert_val_path=None, type='KD'):
        if type=='KD':
            print('Data preparation for KD')
            vocab2id = {w: i for i, w in enumerate(vocab)}
            vec = CountVectorizer(
                vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
            entire_dataset = train.copy()
            if test is not None:
                entire_dataset.extend(test)
            if validation is not None:
                entire_dataset.extend(validation)

            vec.fit(entire_dataset)
            idx2token = {v: k for (k, v) in vec.vocabulary_.items()}

            x_train = vec.transform(train)
            Sb_train = CTMKD.load_bert_data(Sbert_train_path, train, Sbert_model, type='Student')
            Tb_train = CTMKD.load_bert_data(Tbert_train_path, train, Tbert_model, type='Teacher')

            train_data = dataset.CTMDatasetExtended(x_train.toarray(), Sb_train, idx2token, Tb_train)
            input_size = len(idx2token.keys())

            if test is not None and validation is not None:
                x_test = vec.transform(test)
                Sb_test = CTMKD.load_bert_data(Sbert_test_path, test, Sbert_model, type='Student')
                Tb_test = CTMKD.load_bert_data(Tbert_test_path, test, Tbert_model, type='Teacher')
                test_data = dataset.CTMDatasetExtended(x_test.toarray(), Sb_test, idx2token, Tb_test)

                x_valid = vec.transform(validation)
                Sb_val = CTMKD.load_bert_data(Sbert_val_path, validation, Sbert_model, type='Student')
                Tb_val = CTMKD.load_bert_data(Tbert_val_path, validation, Tbert_model, type='Teacher')
                valid_data = dataset.CTMDatasetExtended(x_valid.toarray(), Sb_val, idx2token, Tb_val)
                return train_data, test_data, valid_data, input_size
            if test is None and validation is not None:
                x_valid = vec.transform(validation)
                Sb_val = CTMKD.load_bert_data(Sbert_val_path, validation, Sbert_model, type='Student')
                Tb_val = CTMKD.load_bert_data(Tbert_val_path, validation, Tbert_model, type='Teacher')
                valid_data = dataset.CTMDatasetExtended(x_valid.toarray(), Sb_val, idx2token, Tb_val)
                return train_data, valid_data, input_size
            if test is not None and validation is None:
                x_test = vec.transform(test)
                Sb_test = CTMKD.load_bert_data(Sbert_test_path, test, Sbert_model, type='Student')
                Tb_test = CTMKD.load_bert_data(Tbert_test_path, test, Tbert_model, type='Teacher')
                test_data = dataset.CTMDatasetExtended(x_test.toarray(), Sb_test, idx2token, Tb_test)
                return train_data, test_data, input_size
            if test is None and validation is None:
                return train_data, input_size
        elif type=='Teacher_only':
            print('Data preparation for Teacher model')
            vocab2id = {w: i for i, w in enumerate(vocab)}
            vec = CountVectorizer(
                vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
            entire_dataset = train.copy()
            if test is not None:
                entire_dataset.extend(test)
            if validation is not None:
                entire_dataset.extend(validation)

            vec.fit(entire_dataset)
            idx2token = {v: k for (k, v) in vec.vocabulary_.items()}

            x_train = vec.transform(train)
            b_train = CTMKD.load_bert_data(Tbert_train_path, train, Tbert_model, type='Teacher')

            train_data = dataset.CTMDataset(x_train.toarray(), b_train, idx2token)
            input_size = len(idx2token.keys())

            if test is not None and validation is not None:
                x_test = vec.transform(test)
                b_test = CTMKD.load_bert_data(Tbert_test_path, test, Tbert_model, type='Teacher')
                test_data = dataset.CTMDataset(x_test.toarray(), b_test, idx2token)

                x_valid = vec.transform(validation)
                b_val = CTMKD.load_bert_data(Tbert_val_path, validation, Tbert_model, type='Teacher')
                valid_data = dataset.CTMDataset(x_valid.toarray(), b_val, idx2token)
                return train_data, test_data, valid_data, input_size
            if test is None and validation is not None:
                x_valid = vec.transform(validation)
                b_val = CTMKD.load_bert_data(Tbert_val_path, validation, Tbert_model, type='Teacher')
                valid_data = dataset.CTMDataset(x_valid.toarray(), b_val, idx2token)
                return train_data, valid_data, input_size
            if test is not None and validation is None:
                x_test = vec.transform(test)
                b_test = CTMKD.load_bert_data(Tbert_test_path, test, Tbert_model, type='Teacher')
                test_data = dataset.CTMDataset(x_test.toarray(), b_test, idx2token)
                return train_data, test_data, input_size
            if test is None and validation is None:
                return train_data, input_size

    @staticmethod
    def load_bert_data(bert_path, texts, bert_model, type=None):
        print('{} Bert Model: {}'.format(type, bert_model))
        if bert_path is not None:
            if os.path.exists(bert_path):
                bert_ouput = pkl.load(open(bert_path, 'rb'))
            else:
                bert_ouput = bert_embeddings_from_list(texts, bert_model)
                pkl.dump(bert_ouput, open(bert_path, 'wb'))
        else:
            bert_ouput = bert_embeddings_from_list(texts, bert_model)
        return bert_ouput
