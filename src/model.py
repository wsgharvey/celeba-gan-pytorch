import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from dcgan import Generator
from utils import ensure_batched


def contains_row(tensor, row):
    return torch.any(torch.all(torch.eq(row, tensor), dim=-1))


num_iters = 0


class ObserveGrid():
    def __init__(self, zooms=[2, 4, 8], sizes=[(8, 8), (8, 8), (8, 8)],
                 cuda=True):
        """
        provides method to get foveal views of images

        zooms - how zoomed in each view is:
                e.g. zoom=1: full image
                     zoom=2: twice as "close up"
        sizes - each view is `size[0] x size[1]` pixels
        """
        self.cuda = cuda
        self.zooms = zooms
        self.sizes = sizes
        self.xc = None
        self.yc = None

        grids = self._make_centred_grids()
        # flatten grids since we don't need spatial information
        grids = torch.cat([grid.view(-1, 2)
                           for grid in grids],
                          dim=0)
        # remove duplicates for points that were in overlapping views
        grids = torch.cat([row for i, row in enumerate(grids)
                           if not contains_row(grids[i+1:], row)],
                          dim=0)
        grid = grids.view(1, 1, -1, 2)
        if self.cuda:
            self.grid = grid.cuda()
        else:
            self.grid = grid

    def _make_centred_grids(self):
        identity = torch.Tensor([[[1., 0., 0.],
                                  [0., 1., 0.]]])
        sizes = [(1, 1, size[0], size[1]) for size in self.sizes]
        grids = [F.affine_grid(identity/zoom, size)
                 for zoom, size in zip(self.zooms, sizes)]
        if self.cuda:
            grids = [g.cuda() for g in grids]
        return grids

    def peek(self, images):
        """
        xc, yc \in [1, 1] are coordinates of centre of
        observed patch. currently must be same for every
        image.
        """
        images, (N, C, H, W) = ensure_batched(images)
        batch_grid = self.moved_grid.repeat(N, 1, 1, 1)
        foveal_view = F.grid_sample(images, batch_grid,
                                    padding_mode='zeros')
        return foveal_view.view(N, C, -1)

    def set_pos(self, xc, yc):
        self.xc = xc
        self.yc = yc
        self.pos = torch.Tensor([xc, yc])
        if self.cuda:
            self.pos = self.pos.cuda()
        self.moved_grid = self.grid + self.pos

    def visualise_grid(self, images):
        images, (_, C, H, W) = ensure_batched(images)
        image = images[0].clone()
        grid = self.moved_grid.view(-1, 2)
        for x, y in grid:
            # scale coords to [0, 1]
            x = (x+1)/2
            y = (y+1)/2
            # scale coords to [0, H], [0, W]
            y = int(y*H)
            x = int(x*W)
            if y >= 0 and x >= 0 and y < H and x < W:
                image[:, y, x] = torch.Tensor([1., -1., -1.])
        return image

    def visualise_peeks(self, images):
        """
        generator of images representing what each
        grid `sees`
        """
        images, (_, C, H, W) = ensure_batched(images)
        image = images[0:1]
        grids = [grid + self.pos for grid in self._make_centred_grids()]
        for grid in grids:
            yield F.grid_sample(image, grid,
                                padding_mode='zeros').squeeze(dim=0)

    def copy(self):
        other = ObserveGrid(self.zooms, self.sizes, self.cuda)
        if self.xc is not None:
            other.set_pos(self.xc, self.yc)
        return other


class FaceModel():
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.latent_dim = 100
        self.latent_mean = torch.zeros(1, self.latent_dim)
        self.latent_std = torch.ones(1, self.latent_dim)
        self.obs_std = torch.tensor(1.)
        if cuda:
            self.latent_mean = self.latent_mean.cuda()
            self.latent_std = self.latent_std.cuda()
            self.obs_std = self.obs_std.cuda()
            self.generator = Generator().cuda()
        else:
            self.generator = Generator()
        self.generator.eval()

        self.generator.load_state_dict(
            torch.load('checkpoints/trained_wgan/wgan-gen.pt',
                       map_location=(None if cuda else 'cpu'))
        )

    def __call__(self, observe_grids, observations):
        global num_iters
        num_iters += 1
        latents = pyro.sample(
            "latents",
            dist.Normal(self.latent_mean,
                        self.latent_std)
        )
        image = self.generator(latents).squeeze(0)

        if observe_grids is not None:
            for i, (observe_grid, observation) in enumerate(
                                                      zip(observe_grids,
                                                          observations)):
                sim_foveal = observe_grid.peek(image)
                pyro.sample(
                    f"observed_patch_{i}",
                    dist.Normal(sim_foveal.contiguous().view(-1),
                                self.obs_std),
                    obs=observation.view(-1)
                )
        return image

    def latent_var(self, trace):
        """
        Some arbitrary binary latent variable. Stand-in since
        we don't currently have disentangled representations.
        """
        return trace.nodes['latents']['value'][0, 0] > 0


if __name__ == '__main__':
    from utils import to_pil

    # make an array of samples from the model
    model = FaceModel(False)
    grid = [[model(None, None)
             for _ in range(10)]
            for _ in range(10)]
    grid = torch.cat([torch.cat(row, dim=2) for row in grid],
                     dim=1)
    to_pil(grid).show()

    # vary a single dimension
    dim = 0
    model = FaceModel(False)
    latents = torch.distributions.Normal(
        torch.zeros(20, 100),
        torch.ones(20, 100)
    ).sample()
    hi_latents = latents.clone()
    hi_latents[:, dim] = 1
    lo_latents = latents.clone()
    lo_latents[:, dim] = -1
    hi_images = model.generator(hi_latents)
    lo_images = model.generator(lo_latents)
    hi_row = torch.cat([im for im in hi_images], dim=2)
    lo_row = torch.cat([im for im in lo_images], dim=2)
    grid = torch.cat([hi_row, lo_row], dim=1)
    to_pil(grid).show()
