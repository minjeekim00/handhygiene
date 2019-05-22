from torch.utils.data.sampler import Sampler
import torch as th
import math
import random
from tqdm import tqdm 

def th_random_choice(a, n_samples=1, replace=True, p=None):
    """
    Parameters
    -----------
    a : 1-D array-like
        If a th.Tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a was th.range(n)
    n_samples : int, optional
        Number of samples to draw. Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    Returns
    --------
    samples : 1-D ndarray, shape (size,)
        The generated random samples
    """
    if isinstance(a, int):
        a = th.arange(0, a)

    if p is None:
        if replace:
            idx = th.floor(th.rand(n_samples)*a.size(0)).long()
        else:
            idx = th.randperm(len(a))[:n_samples]
    else:
        if abs(1.0-sum(p)) > 1e-3:
            raise ValueError('p must sum to 1.0')
        if not replace:
            raise ValueError('replace must equal true if probabilities given')
        idx_vec = th.cat([th.zeros(round(p[i]*1000))+i for i in range(len(p))])
        idx = (th.floor(th.rand(n_samples)*999)).long()
        idx = idx_vec[idx].long()
    selection = a[idx]
    if n_samples == 1:
        selection = selection[0]
    return selection

class MultiSampler(Sampler):
    """
        Samples elements more than once in a single pass through the data.
        This allows the number of samples per epoch to be larger than the number
        of samples itself, which can be useful when training on 2D slices taken
        from 3D images, for instance.
    """
    def __init__(self, dataset, desired_samples, shuffle=False):
        """
            Initialize MultiSampler
            Arguments
            ---------
            data_source : the dataset to sample from

            desired_samples : number of samples per batch you want
                whatever the difference is between an even division will
                be randomly selected from the samples.
                e.g. if len(data_source) = 3 and desired_samples = 4, then
                all 3 samples will be included and the last sample will be
                randomly chosen from the 3 original samples.
            shuffle : boolean
                whether to shuffle the indices or not

            Example:
                >>> m = MultiSampler(2, 6)
                >>> x = m.gen_sample_array()
                >>> print(x) # [0,1,0,1,0,1]
        """
        self.dataset = dataset
        self.desired_samples = desired_samples
        self.shuffle = shuffle

    def gen_sample_array(self):
        n_repeats = self.desired_samples / len(self.dataset)
        sample_list = []
        for i in tqdm(range(math.floor(n_repeats))):
            print("i = {}".format(i))
            print("sample list length = {}".format(len(sample_list)))
            for j in tqdm(range(len(self.dataset))):
                print("j = {}".format(j))
                clips, flows, labels = self.dataset[j]
                if j == 0:
                    print(clips.shape, flows.shape, labels.shape)
                sample_list.append(self.dataset[j])
        # add the left over samples
        
        left_over = self.desired_samples % len(self.dataset)
        if left_over > 0:
            i = random.randint(0, len(self.dataset)+1)
            sample_list.append(self.dataset[i])
        self.samples = tuple(sample_list)
        
        print(self.samples)
        return self.samples

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.desired_samples

