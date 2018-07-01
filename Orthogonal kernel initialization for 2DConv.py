import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random

###############################random seed##############################################################################
manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ",manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#torch.cuda.manual_seed_all(manualSeed)
cudnn.benchmark = True
########################################################################################################################


######################################Generating 2D orthogonal initialization kernel####################################
#generating uniform orthogonal matrix
def _orthogonal_matrix(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q

#generating orthogonal projection matrix,i.e. the P,Q of Algorithm1 in the original
def _symmetric_projection(n):
    """Compute a n x n symmetric projection matrix.
    Args:
      n: Dimension.
    Returns:
      A n x n orthogonal projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
    q = _orthogonal_matrix(n)
    # randomly zeroing out some columns
    # mask = math.cast(random_ops.random_normal([n], seed=self.seed) > 0,
    # #                      self.dtype)
    mask = torch.randn(n)

    c = torch.mul(mask,q)
    U,_,_= torch.svd(c)
    U1 = U[:,0].view(len(U[:,0]),1)
    P = torch.mm(U1,U1.t())
    P_orth_pro_mat = torch.eye(n)-P
    return P_orth_pro_mat

#generating block matrix the step2 of the Algorithm1 in the original
def _block_orth(p1, p2):
    """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
    Args:
      p1: A symmetric projection matrix (Square).
      p2: A symmetric projection matrix (Square).
    Returns:
      A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                      [(1-p1)p2, (1-p1)(1-p2)]].
    Raises:
      ValueError: If the dimensions of p1 and p2 are different.
    """
    if p1.shape != p2.shape:
        raise ValueError("The dimension of the matrices must be the same.")
    kernel2x2 = {}#Block matrices are contained by a dictionary
    eye = torch.eye(p1.shape[0])
    kernel2x2[0, 0] = torch.mm(p1, p2)
    kernel2x2[0, 1] = torch.mm(p1, (eye - p2))
    kernel2x2[1, 0] = torch.mm((eye - p1), p2)
    kernel2x2[1, 1] = torch.mm((eye - p1), (eye - p2))

    return kernel2x2

#compute convolution operator of equation2.17 in the original
def _matrix_conv(m1, m2):
    """Matrix convolution.
    Args:
      m1: A k x k dictionary, each element is a n x n matrix.
      m2: A l x l dictionary, each element is a n x n matrix.
    Returns:
      (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """

    n = m1[0, 0].shape[0]
    if n != m2[0, 0].shape[0]:
        raise ValueError("The entries in matrices m1 and m2 "
                         "must have the same dimensions!")
    k = int(np.sqrt(len(m1)))
    l = int(np.sqrt(len(m2)))
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
        for j in range(size):
            result[i, j] = torch.zeros(n,n)
            for index1 in range(min(k, i + 1)):
                for index2 in range(min(k, j + 1)):
                    if (i - index1) < l and (j - index2) < l:
                        result[i, j] += torch.mm(m1[index1, index2],
                                                        m2[i - index1, j - index2])
    return result

def _dict_to_tensor(x, k1, k2):
    """Convert a dictionary to a tensor.
    Args:
      x: A k1 * k2 dictionary.
      k1: First dimension of x.
      k2: Second dimension of x.
    Returns:
      A k1 * k2 tensor.
    """
    return torch.stack([torch.stack([x[i, j] for j in range(k2)])
                            for i in range(k1)])

#generating a random 2D orthogonal Convolution kernel
def _orthogonal_kernel(ksize, cin, cout):
    """Construct orthogonal kernel for convolution.
    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.
    Returns:
      An [ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
    if cin > cout:
        raise ValueError("The number of input channels cannot exceed "
                         "the number of output channels.")
    orth = _orthogonal_matrix(cout)[0:cin, :]#这就是算法1中的H
    if ksize == 1:
        return torch.unsqueeze(torch.unsqueeze(orth,0),0)

    p = _block_orth(_symmetric_projection(cout),
                         _symmetric_projection(cout))
    for _ in range(ksize - 2):
        temp = _block_orth(_symmetric_projection(cout),
                                _symmetric_projection(cout))
        p = _matrix_conv(p, temp)
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = torch.mm(orth, p[i, j])

    return _dict_to_tensor(p, ksize, ksize).permute(3,2,1,0)

#Call method
def Conv2d_weights_orth_init(net, kernel_size=3, in_channels=3, out_channels=64):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data=_orthogonal_kernel(kernel_size,in_channels,out_channels)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
           # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()
'''
for example,
net=nn.Conv2d(3,64,3,2,1)
net.apply(Conv2d_weights_orth_init)
'''

def makeDeltaOrthogonal(in_channels=3, out_channels=64, kernel_size=3, gain=torch.Tensor([1])):
    weights = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
    out_channels = weights.size(0)
    in_channels = weights.size(1)
    if weights.size(1) > weights.size(0):
        raise ValueError("In_filters cannot be greater than out_filters.")
    q = _orthogonal_matrix(out_channels)
    q = q[:in_channels, :]
    q *= torch.sqrt(gain)
    beta1 = weights.size(2) // 2
    beta2 = weights.size(3) // 2
    weights[:, :, beta1, beta2] = q
    return weights
#Calling method is the same as the above _orthogonal_kernel
######################################################END###############################################################






