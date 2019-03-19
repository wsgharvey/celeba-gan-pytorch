import torch
from dcgan import Generator
import time

CUDA_AVAILABLE = torch.cuda.is_available()


# batching gives ~200x speed-up on GPU for N=1000

def tb(N):
    latents = torch.distributions.Normal(torch.zeros(1, 100), torch.ones(1, 100)).sample()
    g = Generator()
    if CUDA_AVAILABLE:
        g = g.cuda()
        latents = latents.cuda()

    start = time.time()
    [g(latents.repeat(2, 1)) for _ in range(int(N/2))]
    print(f"{(time.time()-start)/N}s per trace with no batching")

    start = time.time()
    g(latents.repeat(N, 1))
    print(f"{(time.time()-start)/N}s per trace with batching")
