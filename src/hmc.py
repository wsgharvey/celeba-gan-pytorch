import torch
from pyro.infer.mcmc import MCMC, NUTS

from model import FaceModel, Observer
from utils import to_pil


CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA: ", CUDA_AVAILABLE)

model = FaceModel(cuda=CUDA_AVAILABLE)
nuts_kernel = NUTS(model, adapt_step_size=True)
ground_truth_image = model(None, None)
observer = Observer(cuda=CUDA_AVAILABLE)
observer.set_pos(-0.5, 0.2)

observe_img = observer.visualise_grid(ground_truth_image)
to_pil(observe_img).save("ground_truth.png")


hmc_posterior = MCMC(nuts_kernel, num_samples=10, warmup_steps=0) \
    .run(observer, ground_truth_image)

for i, trace in enumerate(hmc_posterior.exec_traces):
    sampled_img = trace.nodes['_RETURN']['value']
    sampled_img = observer.visualise_grid(sampled_img)
    to_pil(sampled_img).save(f"sample_{i}.png")
