import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Complex(object):
    ''' Simple Complex Number Class for pytorch '''
    def __init__(self, real, imag=None):
        ''' if imag is none we divide real --> [re, im]'''
        if imag is not None:
            assert real.size() == imag.size(), "{}re != {}im".format(
                real.size(), imag.size())
            self._real = real
            self._imag = imag
        else:
            assert real.size(-1) % 2 == 0, "need to be div by two"
            assert real.dim() == 2, "only 2d supported"
            half = real.size(-1) // 2
            self._real = real[:, 0:half]
            self._imag = real[:, half:]

    def unstack(self):
        return torch.cat([self._real, self._imag], dim=-1)

    def __add__(self, other):
        real = self._real + other._real
        imag = self._imag + other._imag
        return Complex(real, imag)

    def __sub__(self, other):
        real = self._real - other._real
        imag = self._imag - other._imag
        return Complex(real, imag)

    def __mul__(self, other):
        real = self._real * other._real + self._imag * other._imag
        imag = self._real * other._imag + self._imag * other._real
        return Complex(real, imag)

    def __rmul__(self, other):
        real = other._real * self._real + other._imag * self._imag
        imag = other._imag * self._real + other._real * self._imag
        return Complex(real, imag)

    def abs(self):
        return torch.sqrt(self._real * self._real + self._imag * self._imag)

    def conj(self):
        return Complex(self._real, -self._imag)

    def size(self):
        return self._real.size()

    def real(self):
        return self._real

    def imag(self):
        return self._imag


def long_type(use_cuda):
    return torch.cuda.LongTensor if use_cuda else torch.LongTensor


def float_type(use_cuda):
    return torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def one_hot(num_cols, indices, use_cuda=False):
    """ Creates a matrix of one hot vectors.
        - num_cols: int
        - indices: FloatTensor array
    """
    batch_size = indices.size(0)
    mask = long_type(use_cuda)(batch_size, num_cols).fill_(0)
    ones = 1
    if isinstance(indices, Variable):
        ones = Variable(long_type(use_cuda)(indices.size()).fill_(1))
        mask = Variable(mask, volatile=indices.volatile)

    return mask.scatter_(1, indices, ones)


def circular_convolution_conv(keys, values, cuda=False):
    '''
    For the circular convolution of x and y to be equivalent,
    you must pad the vectors with zeros to length at least N + L - 1
    before you take the DFT. After you invert the product of the
    DFTs, retain only the first N + L - 1 elements.
    '''
    assert values.dim() == keys.dim() == 2, "only 2 dims supported"
    batch_size = keys.size(0)
    keys_feature_size = keys.size(1)
    values_feature_size = values.size(1)
    required_size = keys_feature_size + values_feature_size - 1

    # zero pad upto N+L-1
    zero_for_keys = Variable(float_type(cuda)(
        batch_size, required_size - keys_feature_size).zero_())
    zero_for_values = Variable(float_type(cuda)(
        batch_size, required_size - values_feature_size).zero_())
    keys = torch.cat([keys, zero_for_keys], -1)
    values = torch.cat([values, zero_for_values], -1)

    # do the conv and reshape and return
    print('values = ', values.view(batch_size, 1, -1).size(), ' keys = ', keys.view(batch_size, 1, -1).size())
    print('conv = ', F.conv1d(values.view(batch_size, 1, -1),
                    keys.view(batch_size, 1, -1)).size())
    return F.conv1d(values.view(batch_size, 1, -1),
                    keys.view(batch_size, 1, -1)).squeeze()[:, 0:required_size]


def circular_convolution_fft(keys, values, normalized=True, conj=False, cuda=False):
    '''
    For the circular convolution of x and y to be equivalent,
    you must pad the vectors with zeros to length at least N + L - 1
    before you take the DFT. After you invert the product of the
    DFTs, retain only the first N + L - 1 elements.
    '''
    assert values.dim() == keys.dim() == 2, "only 2 dims supported"
    assert values.size(-1) % 2 == keys.size(-1) % 2 == 0, "need last dim to be divisible by 2"
    keys_feature_size = keys.size(1)
    values_feature_size = values.size(1)
    required_size = keys_feature_size + values_feature_size - 1
    required_size = required_size + 1 if required_size % 2 != 0 else required_size

    # conj transpose
    keys = Complex(keys).conj().unstack() if conj else keys

    # reshape to [batch, [real, imag]]
    half = keys.size(-1) // 2
    keys = torch.cat([keys[:, 0:half].unsqueeze(2), keys[:, half:].unsqueeze(2)], -1)
    values = torch.cat([values[:, 0:half].unsqueeze(2), values[:, half:].unsqueeze(2)], -1)

    # do the fft, ifft and return num_required
    kf = torch.fft(keys, signal_ndim=1, normalized=normalized)
    vf = torch.fft(values, signal_ndim=1, normalized=normalized)
    kvif = torch.ifft(kf*vf, signal_ndim=1, normalized=normalized)#[:, 0:required_size]

    # if conj:
    #     return Complex(kvif[:, :, 1], kvif[:, :, 0]).unstack()

    return Complex(kvif[:, :, 0], kvif[:, :, 1]).unstack()

    #return Complex(kvif[:, :, 0], kvif[:, :, 1]).abs() if not conj \
    return Complex(kvif[:, :, 0], kvif[:, :, 1]).unstack() # if not conj \
        # else Complex(kvif[:, :, 1], kvif[:, :, 0]).abs()


class HolographicMemory(nn.Module):
    def __init__(self, num_init_memories, normalization='complex', cuda=True):
        super(HolographicMemory, self).__init__()
        self.perms, self.inv_perms, self.memories = None, None, None
        self.num_memories = num_init_memories
        self.complex_normalize = normalization == 'complex'
        self.l2_normalize = normalization == 'l2'
        self.conv_fn = circular_convolution_fft
        self.cuda = cuda

    @staticmethod
    def _generate_perms_and_inverses(feature_size, num_perms):
        perms = [torch.randperm(feature_size)
                 for _ in range(num_perms)]
        inv_perms = [torch.cat([(perm == i).nonzero()
                                for i in range(feature_size)], 0).squeeze()
                     for perm in perms]
        return perms, inv_perms

    def normalize(self, arr):
        if self.complex_normalize:
            return self._complex_normalize(arr)

        return F.normalize(arr, dim=-1)

    def _complex_normalize(self, arr):
        assert arr.size(-1) % 2 == 0, "dim[-1] need to be divisible by 2"
        half = arr.size(-1) // 2
        cplx = Complex(arr[:, 0:half], arr[:, half:]).abs()
        mag = torch.max(cplx, torch.ones_like(cplx))
        return arr / torch.cat([mag, mag], -1)

    def encode(self, keys, values):
        '''
        Encoders some keys and values together

        values: [batch_size, feature_size]
        keys:   [batch_size, feature_size]

        sets memories: [num_memories, features]
        '''
        assert values.dim() == keys.dim() == 2, "only operate over 2 dims"
        batch_size, feature_size = list(values.size())
        if self.perms is None:
            ''' initial generation of random perms '''
            self.perms, self.inv_perms = self._generate_perms_and_inverses(
                feature_size, self.num_memories
            )

        keys = self.normalize(keys)
        permed_keys = torch.cat([keys[:, perm] for perm in self.perms], 0)
        conv_output = self.conv_fn(permed_keys,
                                   values.repeat([self.num_memories, 1]),
                                   cuda=self.cuda)
        self.memories = self.memories + conv_output if self.memories is not None else conv_output

    def extend_memory(self, batch_size, feature_size, num_to_extend):
        if num_to_extend < 1:
            return

        new_perms, new_inv_perms = self._generate_perms_and_inverses(
            feature_size, num_to_extend
        )
        self.perms.extend(new_perms)
        self.inv_perms.extend(new_inv_perms)
        if self.memories is not None:
            zero_vectors = float_type(self.cuda)(batch_size*num_to_extend, feature_size).zero_()
            self.memories = torch.cat([self.memories, zero_vectors], 0)

        self.num_memories += num_to_extend

    def decode(self, keys):
        '''
        Decoders values out of memories

        keys:     [batch_size, feature_size]
        returns: [batch, features]
        '''
        keys = self.normalize(keys)
        batch_size = keys.size(0)

        # re-gather keys to avoid mixing between different keys.
        permed_keys = torch.cat([keys[:, perm] for perm in self.perms], 0)
        unsplit_conv = self.conv_fn(permed_keys, self.memories, conj=False, cuda=self.cuda)
        indices = [[i for i in range(j, self.num_memories*batch_size, batch_size)]
                   for j in range(batch_size)]
        return torch.cat([torch.sum(unsplit_conv[ind], 0) for ind in indices], 0)

if __name__ == "__main__":
    # simple test on MNIST recovery
    import argparse
    import torchvision
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(description='HolographicMemory MNIST Recovery')

    # Task parameters
    parser.add_argument('--key-type', type=str, default='gaussian',
                        help="type of key: gaussian or onehot (default: gaussian)")
    parser.add_argument('--batch-size', type=int, default=10,
                        help="batch size (default: 10)")
    parser.add_argument('--batches-to-encode', type=int, default=10,
                        help="how many minibatches to encode (default: 10)")
    parser.add_argument('--num-memories', type=int, default=10,
                        help="number of memory traces (default: 10)")
    parser.add_argument('--increment-memories-per-batch', type=int, default=0,
                        help="number of memory traces to increase per batch (default: 0)")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    feature_size =  784
    mnist = torch.utils.data.DataLoader(
        datasets.MNIST('.datasets', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
    )

    # build memory and some random keys
    memory = HolographicMemory(num_init_memories=args.num_memories,
                               normalization='complex', cuda=args.cuda)
    if args.key_type == 'gaussian':
        keys = [torch.randn(args.batch_size, feature_size)
                for _ in range(args.batches_to_encode)]
    else:
        rv = torch.distributions.OneHotCategorical(probs=torch.rand(args.batch_size, feature_size))
        keys = [rv.sample() for _ in range(args.batches_to_encode)]

    if args.cuda:
        keys = [k.cuda() for k in keys]

    # encode some images
    img_container, key_container = [], []
    for i, (img, lbl) in enumerate(mnist):
        if i > args.batches_to_encode - 1:
            break

        img, lbl = img.cuda() if args.cuda else img, lbl.cuda() if args.cuda else lbl
        img_container.append(img)
        memory.encode(keys[i], img.view(args.batch_size, -1))
        # lbl = lbl.unsqueeze(1) if lbl.dim() < 2 else lbl
        # key_container.append(one_hot(feature_size, lbl, True).type(float_type(True)))
        # print(img.size(), lbl.size(), key_container[-1].size())
        # memory.encode(key_container[-1], img.view(args.batch_size, -1))

        # expand_mem if requested
        memory.extend_memory(args.batch_size, feature_size, args.increment_memories_per_batch)

    img_container = torch.cat(img_container, 0)
    # keys = torch.cat(key_container, 0)
    # print("key container post = ", keys.size())
    print("encoded {} samples x {} --> {}".format(
        args.batch_size, list(img.size()), list(memory.memories.size())))

    # try to decode
    values = torch.cat([memory.decode(key) for key in keys], 0)
    print("decoded {} keys --> {}".format(
        list(torch.cat(keys, 0).size()), values.size()))


    # save image for visualization
    grid = torchvision.utils.make_grid(
        torch.cat([img_container, values.view(-1, 1, 28, 28)], 0),
        nrow=args.batch_size, normalize=True, scale_each=True
    )
    def show(img):
        import matplotlib.pyplot as plt
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.show()

    show(grid)
