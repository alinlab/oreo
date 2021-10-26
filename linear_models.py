import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import AddCoords


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


## For CCIL
class CoordConvEncoder(nn.Module):
    def __init__(self, input_channels, embedding_dim, ch_div=1):
        super(CoordConvEncoder, self).__init__()
        self.coordconv = AddCoords(2, with_r=False, use_cuda=True)
        self.conv1 = nn.Conv2d(input_channels + 2, 64 // ch_div, 5, stride=2)
        self.conv2 = nn.Conv2d(64 // ch_div, 128 // ch_div, 5, stride=2)
        self.conv3 = nn.Conv2d(128 // ch_div, 256 // ch_div, 5, stride=2)
        self.conv4 = nn.Conv2d(256 // ch_div, 512 // ch_div, 5, stride=2)

        self.fc = nn.Linear(512 // ch_div, embedding_dim)

    def forward(self, x):
        x = self.coordconv(x)  # Bx3x64x64
        x = F.leaky_relu(self.conv1(x), 0.2)  # Bx64x30x30
        x = F.leaky_relu(self.conv2(x), 0.2)  # Bx128x13x13
        x = F.leaky_relu(self.conv3(x), 0.2)  # Bx256x5x5
        x = F.leaky_relu(self.conv4(x), 0.2)  # Bx512x1x1

        x = self.fc(torch.flatten(x, start_dim=1))  # BxN
        return x


class CoordConvDecoder(nn.Module):
    def __init__(self, input_channels, embedding_dim, ch_div=1):
        super(CoordConvDecoder, self).__init__()
        self.coordconv = AddCoords(2, with_r=False, use_cuda=True)
        self.conv1 = nn.Conv2d(embedding_dim + 2, 512 // ch_div, 1)
        self.conv2 = nn.Conv2d(512 // ch_div, 256 // ch_div, 1)
        self.conv3 = nn.Conv2d(256 // ch_div, 256 // ch_div, 1)
        self.conv4 = nn.Conv2d(256 // ch_div, 128 // ch_div, 1)
        self.conv5 = nn.Conv2d(128 // ch_div, 64 // ch_div, 1)
        self.conv6 = nn.Conv2d(64 // ch_div, input_channels, 1)

    def forward(self, x):
        x = x.view(-1, x.shape[1], 1, 1)  # BxNx1x1
        x = x.repeat(1, 1, 64, 64)  # BxNx64x64
        x = self.coordconv(x)  # Bx(N+2)x64x64
        x = F.relu(self.conv1(x))  # Bx512x64x64
        x = F.relu(self.conv2(x))  # Bx256x64x64
        x = F.relu(self.conv3(x))  # Bx256x64x64
        x = F.relu(self.conv4(x))  # Bx128x64x64
        x = F.relu(self.conv5(x))  # Bx64x64x64

        x = torch.sigmoid(self.conv6(x))  # Bx1x64x64
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            # + torch.sum(self._embedding.weight**2, dim=1)
            + torch.sum(self._embedding.weight.t() ** 2, dim=0, keepdim=True)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction="none").mean(
            (1, 2, 3)
        )
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction="none").mean(
            (1, 2, 3)
        )
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        return quantized, loss, perplexity, encodings, encoding_indices, distances


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels,
        embedding_dim,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super(Encoder, self).__init__()
        self._input_channels = input_channels
        self._embedding_dim = embedding_dim
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        ## 42 x 42
        self._conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_hiddens // 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 21 x 21
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 4,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 10 x 10
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 8 x 8
        self._conv_4 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        ## 8 x 8
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        ## 8 x 8
        self._conv_5 = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        self.apply(weight_init)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        x = F.relu(x)

        x = self._conv_4(x)
        x = self._residual_stack(x)
        return self._conv_5(x)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        embedding_dim,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super(Decoder, self).__init__()
        self._out_channles = out_channels
        self._embedding_dim = embedding_dim
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        ## 8 x 8
        self._conv_1 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        ## 8 x 8
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        ## 10 x 10
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1,
        )
        ## 21 x 21
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
        )
        ## 42 x 42
        self._conv_trans_3 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens // 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 84 x 84
        self._conv_trans_4 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 4,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.apply(weight_init)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)
        x = F.relu(x)

        x = self._conv_trans_3(x)
        x = F.relu(x)
        return self._conv_trans_4(x)


class VQVAEModel(nn.Module):
    def __init__(self, encoder, decoder, quantizer):
        super(VQVAEModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._quantizer = quantizer

    def forward(self, x, encode_only=False):
        z = self._encoder(x)
        quantized, vq_loss, _, _, encoding_indices, _ = self._quantizer(z)
        if encode_only:
            x_recon = None
        else:
            x_recon = self._decoder(quantized)
        return z, x_recon, vq_loss, quantized, encoding_indices


class CoordConvBetaVAE(nn.Module):
    def __init__(self, z_dim=32, ch_div=1):
        super(CoordConvBetaVAE, self).__init__()

        self.encoder = CoordConvEncoder(1, z_dim * 2, ch_div)
        self.decoder = CoordConvDecoder(1, z_dim, ch_div)

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=-1)
        sigma = torch.exp(logvar / 2.0)
        epsilon = torch.randn_like(mu)
        z = mu + sigma * epsilon
        return z, mu, sigma

    def decode(self, x):
        x = self.decoder(x)
        return x

    ## for dataparallel
    def forward(self, x, mode="encode"):
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode(x)
        else:
            raise NotImplementedError


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class PreActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.trunk = mlp(z_dim, None, out_dim, 0)
        self.apply(weight_init)

    def forward(self, h):
        h = torch.flatten(h, start_dim=1)
        logits = self.trunk(h)
        return logits


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, z_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(z_dim, hidden_dim, action_dim, hidden_depth)
        self.apply(weight_init)

    def forward(self, h):
        logits = self.trunk(h)
        return logits


class Projector(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, z_dim, out_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(z_dim, hidden_dim, out_dim, hidden_depth)
        self.apply(weight_init)

    def forward(self, h):
        outputs = self.trunk(h)
        return outputs
