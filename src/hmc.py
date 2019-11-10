import torch
from pyro.infer.mcmc import MCMC, NUTS

from model import FaceModel, ObserveGrid, num_iters
from utils import visualise_sample


CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA: ", CUDA_AVAILABLE)

model = FaceModel(cuda=CUDA_AVAILABLE)
nuts_kernel = NUTS(model, adapt_step_size=True)
ground_truth_image = model(None, None)
observe_grid = ObserveGrid(zooms=[1, 2, 4],
                           sizes=[(20, 20), (20, 20), (20, 20)],
                           cuda=CUDA_AVAILABLE)
observe_grid.set_pos(-0.5, 0)
observed = observe_grid.peek(ground_truth_image)

# run HMC/NUTS
num_samples = 50
hmc = MCMC(nuts_kernel,
           num_samples=num_samples,
           warmup_steps=num_samples)
hmc.run([observe_grid], [observed])

hmc_samples = {k: v.detach().cpu().numpy()
               for k, v in hmc.get_samples().items()}

# save images
for i, latent in enumerate(hmc._samples['latents'][0]):
    sampled_img = model.generator(latent).squeeze(0)
    visualise_sample(
        ground_truth_image,
        sampled_img,
        observe_grid
    ).save(f"sample_{i}.png")
