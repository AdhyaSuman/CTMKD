import torch
from torch.utils.data import Dataset
import scipy.sparse


class CTMDataset(Dataset):

    """Class to load BOW dataset."""

    def __init__(self, X, X_bert, idx2token):
        """
        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        if X.shape[0] != len(X_bert):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")

        self.X = X
        self.X_bert = X_bert
        self.idx2token = idx2token

    def __len__(self):
        """Return length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if type(self.X[i]) == scipy.sparse.csr.csr_matrix:
            X = torch.FloatTensor(self.X[i].todense())
            X_bert = torch.FloatTensor(self.X_bert[i])
        else:
            X = torch.FloatTensor(self.X[i])
            X_bert = torch.FloatTensor(self.X_bert[i])

        return {'X': X, 'X_bert': X_bert}


class CTMDatasetExtended(CTMDataset):

    """Class to load BoW and the contextualized embeddings."""

    def __init__(self, X, X_bert, idx2token, X_contextual_teacher):
        """
        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        super().__init__(X, X_bert, idx2token)
        self.X_contextual_teacher = X_contextual_teacher


    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        itemDict = super().__getitem__(i)
        itemDict['X_contextual_teacher'] = torch.FloatTensor(self.X_contextual_teacher[i])

        return itemDict