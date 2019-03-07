import torch
import pyro
import pyro.distributions as dist

from dcgan import Generator


class FaceModel():
    def __init__(self):
        self.latent_dim = 100
        self.generator = Generator()
        self.generator.load_state_dict(
            torch.load('checkpoints/trained_wgan/wgan-gen.pt',
                       map_location='cpu')
        )

    def __call__(self, obs_coords, obs_patch):
        latents = pyro.sample(
            "latents",
            dist.Normal(torch.zeros(2, self.latent_dim),
                        torch.ones(2, self.latent_dim))
        ).view(2, self.latent_dim)
        image = self.generator(latents)[0]

        x0, y0 = obs_coords
        sim_patch = image[:, y0:y0+10, x0:x0+10]
        pyro.sample(
            "observed_patch",
            dist.Normal(sim_patch.contiguous().view(-1),
                        torch.tensor(1.)),
            obs=obs_patch.contiguous().view(-1)
        )

        return image
