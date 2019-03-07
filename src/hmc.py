import torch

import pyro
from pyro.infer.mcmc import MCMC, NUTS

from model import FaceModel

CUDA_AVAILABLE = torch.cuda.is_available()

model = FaceModel(cuda=CUDA_AVAILABLE)

nuts_kernel = NUTS(model, adapt_step_size=True)

obs_coords = (10, 15)
obs_patch = torch.tensor([[[ 0.9183,  0.9098,  0.9013,  0.9000,  0.8941,  0.9200,  0.8972,
                             0.7919,  0.3055, -0.5420],
                           [ 0.9180,  0.9042,  0.8985,  0.8930,  0.8748,  0.8798,  0.7352,
                             0.2434, -0.4130, -0.7094],
                           [ 0.9249,  0.9106,  0.8983,  0.8791,  0.8622,  0.7824,  0.2756,
                             -0.3332, -0.5768, -0.7060],
                           [ 0.9246,  0.8956,  0.8798,  0.8379,  0.7683,  0.2672, -0.4072,
                             -0.4930, -0.6180, -0.8200],
                           [ 0.9241,  0.8895,  0.8030,  0.6434,  0.2726, -0.4050, -0.4717,
                             -0.3745, -0.7594, -0.9052],
                           [ 0.8688,  0.5900,  0.1276, -0.1517, -0.3895, -0.5132, -0.3886,
                             -0.6804, -0.8725, -0.9068],
                           [ 0.3307, -0.3829, -0.4690, -0.4063, -0.4765, -0.3377, -0.3993,
                             -0.7935, -0.8924, -0.8734],
                           [-0.4821, -0.5715, -0.4463, -0.4398, -0.4385, -0.3860, -0.6009,
                            -0.7658, -0.8582, -0.8430],
                           [-0.5784, -0.4223, -0.4402, -0.4593, -0.4526, -0.5168, -0.6785,
                            -0.6854, -0.7988, -0.8360],
                           [-0.5096, -0.3975, -0.4538, -0.4962, -0.4949, -0.6749, -0.7084,
                            -0.7482, -0.8453, -0.7262]],

                          [[ 0.9210,  0.9159,  0.9065,  0.9110,  0.9003,  0.9208,  0.8852,
                             0.7923,  0.2568, -0.6433],
                           [ 0.9201,  0.9084,  0.8960,  0.8976,  0.8772,  0.8766,  0.7107,
                             0.1934, -0.5015, -0.7770],
                           [ 0.9200,  0.9145,  0.8930,  0.8871,  0.8650,  0.7668,  0.1926,
                             -0.4448, -0.6349, -0.7867],
                           [ 0.9167,  0.9038,  0.8675,  0.8230,  0.7465,  0.2427, -0.5020,
                             -0.5692, -0.6702, -0.8739],
                           [ 0.9232,  0.8942,  0.7629,  0.5607,  0.1745, -0.5318, -0.5989,
                             -0.4655, -0.8194, -0.9393],
                           [ 0.8658,  0.5397, -0.0858, -0.3546, -0.5345, -0.6580, -0.5533,
                             -0.7834, -0.9245, -0.9464],
                           [ 0.3078, -0.5250, -0.6713, -0.6223, -0.6167, -0.4867, -0.5363,
                             -0.8870, -0.9371, -0.9277],
                           [-0.6325, -0.7398, -0.6191, -0.6052, -0.5974, -0.5121, -0.7205,
                            -0.8479, -0.9163, -0.9120],
                           [-0.7348, -0.5391, -0.6005, -0.6522, -0.5927, -0.6475, -0.7505,
                            -0.7722, -0.8758, -0.9158],
                           [-0.6397, -0.6181, -0.6422, -0.6453, -0.6212, -0.7851, -0.7942,
                            -0.8362, -0.9202, -0.8628]],

                          [[ 0.9189,  0.9135,  0.9039,  0.8880,  0.8793,  0.8995,  0.8781,
                             0.7418,  0.1416, -0.7328],
                           [ 0.9149,  0.9055,  0.8951,  0.8813,  0.8473,  0.8464,  0.6877,
                             0.0360, -0.6189, -0.8483],
                           [ 0.9116,  0.9003,  0.8817,  0.8725,  0.8401,  0.7214,  0.0581,
                             -0.6026, -0.7582, -0.8522],
                           [ 0.9080,  0.8792,  0.8487,  0.7939,  0.7260,  0.1214, -0.5937,
                             -0.6751, -0.7632, -0.9009],
                           [ 0.8997,  0.8411,  0.6986,  0.4699,  0.0330, -0.6073, -0.6659,
                             -0.5837, -0.8690, -0.9593],
                           [ 0.8555,  0.4399, -0.2623, -0.4590, -0.6416, -0.7171, -0.6231,
                             -0.8298, -0.9492, -0.9608],
                           [ 0.2327, -0.5981, -0.7212, -0.6825, -0.7104, -0.5977, -0.6406,
                             -0.9189, -0.9579, -0.9440],
                           [-0.6836, -0.7586, -0.6977, -0.6933, -0.6935, -0.6189, -0.8016,
                            -0.8963, -0.9390, -0.9344],
                           [-0.8005, -0.6583, -0.7008, -0.7350, -0.7162, -0.7748, -0.8457,
                            -0.8570, -0.9209, -0.9446],
                           [-0.7792, -0.6815, -0.7247, -0.7436, -0.7619, -0.8765, -0.8695,
                            -0.8998, -0.9481, -0.9146]]]).cuda()

hmc_posterior = MCMC(nuts_kernel, num_samples=100, warmup_steps=20) \
    .run(obs_coords, obs_patch)
