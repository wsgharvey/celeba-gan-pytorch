import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from dcgan import Generator
from utils import ensure_batched


def contains_row(tensor, row):
    return torch.any(torch.all(torch.eq(row, tensor), dim=-1))


class Observer():
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

        grids = self._make_centred_grids(cuda)
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

    def _make_centred_grids(self, cuda):
        identity = torch.Tensor([[[1., 0., 0.],
                                  [0., 1., 0.]]])
        sizes = [(1, 1, size[0], size[1]) for size in self.sizes]
        grids = [F.affine_grid(identity/zoom, size)
                 for zoom, size in zip(self.zooms, sizes)]
        if cuda:
            grids = grids.cuda()
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

    def __call__(self, observer, true_image):
        latents = pyro.sample(
            "latents",
            dist.Normal(self.latent_dim,
                        self.latent_std)
        ).view(2, self.latent_dim)
        image = self.generator(latents)[0]

        if observer is not None:
            true_foveal = observer.peek(true_image)
            sim_foveal = observer.peek(image)

            pyro.sample(
                "observed_patch",
                dist.Normal(sim_foveal.contiguous().view(-1),
                            self.obs_std),
                obs=true_foveal.contiguous().view(-1)
            )
        return image
