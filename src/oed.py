import torch
from pyro.infer.mcmc import MCMC, NUTS

from model import FaceModel, ObserveGrid
from utils import xlogx


CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA: ", CUDA_AVAILABLE)

model = FaceModel(cuda=CUDA_AVAILABLE)
torch.manual_seed(0)
true_image = model(None, None)

base_observe_grid = ObserveGrid(zooms=[1, 2, 4],
                                sizes=[(20, 20), (20, 20), (20, 20)],
                                cuda=CUDA_AVAILABLE)
possible_designs = [(-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5),
                    (0.0, -0.5),  (0.0, 0.0),  (0.0, 0.5),
                    (0.5, -0.5),  (0.5, 0.0),  (0.5, 0.5)]
N1 = 5
N2 = 5
T = 5


def optimal_sequence(true_image):
    """
    returns sequence of optimal designs (1 at each time step).
    currently only for case when we have not made any previous observations.
    each design (d) is chosen to minimise:
    APE = \mathbb{E}_{p(y|d)} [ \entropy[p(x|y,d)]  ]
    - expectation over p(y|d) is calculated with N1 samples
    - p(x|y,d) is estimated with N2 samples for computing entropy
    """
    designs = []
    observations = []
    observe_grids = []

    for t in range(T):
        # sample N1 images conditioned on previous observations
        # TODO: don't always reinitialise kernel
        nuts_kernel = NUTS(model, adapt_step_size=True)
        posterior = MCMC(
            nuts_kernel,
            num_samples=N1
        ).run(observe_grids, observations)
        possible_images = [t.nodes['_RETURN']['value']
                           for t in posterior.exec_traces]

        # make an ObserveGrid which will be moved to different locations
        observe_grid_t = base_observe_grid.copy()

        # do inference for every design and 'imagined' image we have
        best_APE = -float("inf")
        best_design = None
        for design in possible_designs:
            observe_grid_t.set_pos(*design)
            APE = 0

            for image in possible_images:
                # run HMC to get N2 samples.
                # TODO: don't always reinitialise kernel
                nuts_kernel = NUTS(model, adapt_step_size=True)
                posterior = MCMC(
                    nuts_kernel,
                    num_samples=N2
                ).run(observe_grids+[observe_grid_t],
                      observations+[observe_grid_t.peek(image)])

                # calculate entropy of some latent
                N_true = sum(model.latent_var(trace) for trace
                             in posterior.exec_traces).item()
                p = N_true/N2
                entropy = - xlogx(p) - xlogx(1-p)
                APE += entropy/N2

            if APE > best_APE:
                best_APE = APE
                best_design = design

        # record optimal design
        designs.append(best_design)

        # make observations
        observe_grid_t.set_pos(*best_design)
        observations.append(
            observe_grid_t.peek(true_image)
        )
        observe_grids.append(observe_grid_t)

    return designs
