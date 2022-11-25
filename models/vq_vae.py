import os, os.path
import torch
import numpy as np
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from torch_topological.nn import VietorisRipsComplex, MultiScaleKernel, SignatureLoss, SummaryStatisticLoss, WassersteinDistance


DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
DISABLED = False
BETTI_STEP = torch.tensor(100.0, device=DEVICE)
D_B = torch.tensor(1.0, device=DEVICE)
EPS_STEP = torch.tensor(0.5, device=DEVICE)



hyperp = """
model_params:
  name: 'VQVAE'
  in_channels: 3
  embedding_dim: 64
  num_embeddings: 128
  img_size: 64
  beta: 0.25
"""


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta_num_embeddings: int,
                 beta_embedding_dim: int,
                 is_beta = False,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        if is_beta:
            self.K = beta_num_embeddings
            self.D = beta_embedding_dim
        else:
            self.K = num_embeddings
            self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.checkpoint = lambda: print("unset")
        # initialising embedding space to uniform distribution
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        self.vr1 = []
        self.vr0 = []
        self.vr1loss = 99999
        self.vr0metricloss = 99999
        # k-pass average
        self.wdiff = torch.tensor([], device=DEVICE)
        self.input_wdiff = torch.tensor([], device=DEVICE)
        self.lastwentropy = torch.tensor(0.1, device=DEVICE)
        self.embtropy = torch.tensor(0.1, device=DEVICE)
        self.targ_betti = BETTI_STEP
        self.targ_eps = EPS_STEP
        self.cpass = 1
        self.vq_com_loss = 1
        self.vq_emb_loss = 1
        self.input_entropy = torch.tensor(0.0, device=DEVICE)
        self.eloss = torch.tensor(0.0, device=DEVICE)
        self.vr1_entropy =  torch.tensor(0.0, device=DEVICE)
        self.vr0_entropy = torch.tensor(0.0, device=DEVICE)


    def get_vrs(self):
        ripser = VietorisRipsComplex(dim=1)
        vr = ripser(self.embedding.weight)
        return [self.get_loops(vr, 0), self.get_loops(vr, 1)]

    # original input: one-hot quantized latents
    @staticmethod
    def entropy(x_l_):
        _k_elem = x_l_.sum(dim=-1)
        _k_prob = torch.abs(_k_elem / torch.sum(_k_elem))
        entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8))
        return entropy


    @staticmethod
    def narrow_gaussian(x, ell):
        return torch.exp(-0.5 * (x / ell) ** 2)

    def approx_count_nonzero(self, x, ell=1e-3):
        return len(x) - self.narrow_gaussian(x, ell).sum(dim=-1)


    def get_loops(self, diagrams, dim):
            loop_points = diagrams[dim][1]  # Cohomology *, 1 is diagram, 0 is pairing, 3 is cocycles TBD
            if (dim == 0):
                y_heights = loop_points[:, 1]
                self.vr0_entropy = shannon_entropy(y_heights)
            else:
                y_heights = loop_points[:, 1] - loop_points[:, 0]
                self.vr1_entropy = shannon_entropy(y_heights)
            return [torch.mean(y_heights), self.approx_count_nonzero(y_heights)]

    def entropy_loss(self):
        return torch.abs(D_B - (torch.mean(self.wdiff)/torch.mean(self.input_wdiff)))


    def forward(self, latents: Tensor, validation) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]



        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]



        localentropy = 0
        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        with torch.no_grad():
            l = quantized_latents.detach().clone()
            embtropy = self.entropy(self.embedding.weight.detach().clone())
            localentropy = self.entropy(l)
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]


        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vr = self.get_vrs()
        self.vr0 = vr[0]
        self.vr1 = vr[1]
        vr1_betti = vr[1][1]
        vr0_eps = vr[0][0]

        global DISABLED

        is_ck = (not (self.cpass % 200))

        if ((not (self.cpass % 5000)) or (not (self.cpass % 3000))):
            self.checkpoint("novrrun")

        if not validation:
            try:
                diff = localentropy
                if torch.is_nonzero(diff):
                    self.wdiff = torch.cat([self.wdiff, diff.unsqueeze(0)])
                    self.input_wdiff = torch.cat([self.input_wdiff, self.input_entropy.detach().clone().unsqueeze(0)])
                self.embtropy = embtropy
                self.lastwentropy = localentropy
            except Exception as e:
                self.lastwentropy = localentropy


        if not validation and not DISABLED and is_ck:
            self.eloss = self.entropy_loss()


        if not validation:
            if is_ck:
                self.wdiff = torch.tensor([], device=DEVICE)
                self.input_wdiff = torch.tensor([], device=DEVICE)
            self.cpass = self.cpass + 1


            vr1loss = F.mse_loss(vr1_betti, self.targ_betti)
            vr0metricloss = 10000.0 * F.mse_loss(vr0_eps, self.targ_eps)
            self.vr1loss  = vr1loss
            self.vr0metricloss  = vr0metricloss


        self.vq_com_loss = commitment_loss
        self.vq_emb_loss = self.beta + embedding_loss

        vq_loss = commitment_loss * self.beta + embedding_loss + self.eloss*self.vr1loss + self.eloss*self.vr0metricloss
        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 beta_embedding_dim: int,
                 beta_num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()


        self.save_checkpoint = lambda: print("unset")
        self.embedding_dim = embedding_dim
        self.zzparams = []
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        beta_num_embeddings,
                                        beta_embedding_dim,
                                        False,
                                        self.beta)



        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def checkpoint(self, label):
        logdir = "./logs/VQVAE/"
        v = str(len(os.listdir(logdir)) - 1)
        run = self.vq_layer.cpass
        torch.save(self.zzparams, "saves/v" + v + "run" + str(run) + ".zzparams")
        self.save_checkpoint("saves/v" + v + "run" + str(run) + ".ckpt")
        return True

    def set_save_ck(self, save_checkpoint):
        self.save_checkpoint = save_checkpoint
        self.vq_layer.checkpoint = self.checkpoint

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, disentanglement=False, validation=False, **kwargs) -> List[Tensor]:
        self.vq_layer.input_entropy = self.vq_layer.entropy(input)
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding, validation)
        return [self.decode(quantized_inputs), input, vq_loss]



    def loss_function(self, *args, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss + vq_loss
        pass_zzparams = {"recons_loss": recons_loss,
                         "vq_loss": vq_loss,
                         "vq_com_loss":self.vq_layer.vq_com_loss,
                         "vq_emb_loss":self.vq_layer.vq_emb_loss,
                         "targ_eps": self.vq_layer.targ_eps,
                         "targ_betti": self.vq_layer.targ_betti,
                         "lastwentropy": self.vq_layer.lastwentropy,
                         "inputentropy": self.vq_layer.input_entropy,
                         "vr1loss": self.vq_layer.vr1loss,
                         "vr0metricloss": self.vq_layer.vr0metricloss,
                         "vr1": self.vq_layer.vr1,
                         "vr0": self.vq_layer.vr0,
                         "wdiff": torch.mean(self.vq_layer.wdiff),
                         "input_wdiff": torch.mean(self.vq_layer.input_wdiff),
                         "vr0_entropy": self.vq_layer.vr0_entropy,
                         "vr1_entropy": self.vq_layer.vr1_entropy,
                         "embtropy": self.vq_layer.embtropy,
                         "eloss": self.vq_layer.eloss
                         }
        print(pass_zzparams)
        self.zzparams = [*self.zzparams, pass_zzparams]
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

def shannon_entropy(x_l_):
    _k_elem = x_l_
    _k_prob = torch.abs(_k_elem / torch.sum(_k_elem))
    entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8))
    return entropy
