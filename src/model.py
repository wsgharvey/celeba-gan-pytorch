import torch
import pyro
import pyro.distributions as dist

from dcgan import Generator


class FaceModel():
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.latent_dim = 100
        self.latent_mean = torch.zeros(2, self.latent_dim)
        self.latent_std = torch.ones(2, self.latent_dim)
        self.obs_std = torch.tensor(1.)

        if cuda:
            self.latent_mean = self.latent_mean.cuda()
            self.latent_std = self.latent_std.cuda()
            self.obs_std = self.obs_std.cuda()
            self.generator = Generator().cuda()
        else:
            self.generator = Generator()

        self.generator.load_state_dict(
            torch.load('checkpoints/trained_wgan/wgan-gen.pt',
                       map_location=(None if cuda else 'cpu'))
        )

    def __call__(self, obs_coords, obs_patch):
        latents = pyro.sample(
            "latents",
            dist.Normal(self.latent_dim,
                        self.latent_std)
        ).view(2, self.latent_dim)
        image = self.generator(latents)[0]

        x0, y0 = obs_coords
        sim_patch = image[:, y0:y0+10, x0:x0+10]
        pyro.sample(
            "observed_patch",
            dist.Normal(sim_patch.contiguous().view(-1),
                        self.obs_std),
            obs=obs_patch.contiguous().view(-1)
        )
        return image
