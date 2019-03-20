from math import log

import torch
from pyro.infer.mcmc import MCMC, NUTS

from model import FaceModel, Observer
from utils import xlogx


CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA: ", CUDA_AVAILABLE)

model = FaceModel(cuda=CUDA_AVAILABLE)
observer = Observer(zooms=[1, 2, 4],
                    sizes=[(20, 20), (20, 20), (20, 20)],
                    cuda=CUDA_AVAILABLE)

torch.manual_seed(0)
true_image = model(None, None)

possible_designs = [(-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5),
                    (0.0, -0.5),  (0.0, 0.0),  (0.0, 0.5),
                    (0.5, -0.5),  (0.5, 0.0),  (0.5, 0.5)]
N1 = 5
N2 = 5


def select_design():
    """
    returns optimal design.
    currently only for case when we have not made any previous observations.
    """
    # sample N1 images from model conditioned on all (zero) previous observations
    possible_images = [model(None, None)
                       for _ in range(N1)]

    # do inference for every image and design we have
    best_APE = 0
    best_design = None
    for design in possible_designs:
        APE = 0
        observer.set_pos(*design)
        for image in possible_images:
            # run HMC to get N2 samples
            nuts_kernel = NUTS(model, adapt_step_size=True)
            # TODO: don't reinitialise kernel all the time
            posterior = MCMC(nuts_kernel,
                             num_samples=N2) \
                .run(observer, true_image)
            # calculate entropy of some latent
            N_true = sum(model.latent_var(trace) for trace
                         in posterior.exec_traces)
            print(N_true, N2)
            p = N_true/N2
            entropy = - xlogx(p) - xlogx(1-p)
            APE += entropy/N2

        if APE > best_APE:
            best_APE = APE
            best_design = design

    return best_design
