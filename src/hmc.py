import torch
from pyro.infer.mcmc import MCMC, NUTS

from model import FaceModel, Observer, num_iters
from utils import visualise_sample


CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA: ", CUDA_AVAILABLE)

model = FaceModel(cuda=CUDA_AVAILABLE)
nuts_kernel = NUTS(model, adapt_step_size=True)
ground_truth_image = model(None, None)
observer = Observer(zooms=[1, 2, 4],
                    sizes=[(20, 20), (20, 20), (20, 20)],
                    cuda=CUDA_AVAILABLE)
observer.set_pos(-0.5, 0)

# run HMC/NUTS
num_samples = 50
hmc_posterior = MCMC(nuts_kernel,
                     num_samples=num_samples,
                     warmup_steps=num_samples) \
    .run(observer, ground_truth_image)

print(num_iters, num_samples)

# save images
for i, trace in enumerate(hmc_posterior.exec_traces):
    sampled_img = trace.nodes['_RETURN']['value']
    visualise_sample(
        ground_truth_image,
        sampled_img,
        observer
    ).save(f"sample_{i}.png")
