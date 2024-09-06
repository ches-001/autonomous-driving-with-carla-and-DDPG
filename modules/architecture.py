import math
import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock
from typing import *
    

class ImageEncoder(ResNet):
    def __init__(
        self, 
        in_channels: int, 
        out_features: int,
        block: Union[str, Type]=BasicBlock, 
        block_layers: Optional[Iterable[int]]=None,
        dropout: float=0.0
    ):
        if isinstance(block, str):
            block = getattr(resnet, block)
        super(ImageEncoder, self).__init__(block=block, layers=block_layers or [1, 1, 1, 1])
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self._enc = nn.Sequential(
            nn.Linear(512 if block == BasicBlock else 2048, out_features, bias=False),
            nn.Dropout(dropout)
        )

        #delete unwanted layers
        del self.maxpool, self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype in [torch.uint8, torch.float32]
        if x.dtype == torch.uint8:
            x = (x / 255).to(dtype=torch.float32, device=x.device)
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = output.flatten(start_dim=1, end_dim=-1)
        output = self._enc(output)
        return output
    

class MeasurementEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float=0.0):
        super(MeasurementEncoder, self).__init__()
        self.in_features = in_features
        self.out_features =  out_features

        self._encoder = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._encoder(x)
        return output
    

class ContinuousActionEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float=0.0):
        super(ContinuousActionEncoder, self).__init__()
        self.in_features = in_features
        self.out_features =  out_features

        self._encoder = nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._encoder(x)
        return output
    

class LatentRep(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(LatentRep, self).__init__()
        self.in_features = in_features
        self.out_features =  out_features

        self._encoder = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._encoder(x)
        return output
    
    

class _CommonModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_measurements: int,
            num_intentions: int,
            action_dim: int,
            hidden_dim: int=256,
            img_enc_output_dim: int=512,
            measurement_enc_output_dim: int=128,
            action_enc_output_dim: Optional[int]=None,
            img_enc_dropout: float=0.0,
            measurement_enc_dropout: float=0.0,
            action_enc_dropout: float=0.0,
            **kwargs
        ):
        super(_CommonModule, self).__init__()

        self.in_channels = in_channels
        self.num_intentions = num_intentions
        self.num_measurements = num_measurements
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.img_enc_output_dim = img_enc_output_dim
        self.measurement_enc_output_dim = measurement_enc_output_dim
        self.action_enc_output_dim = action_enc_output_dim

        self._image_encoder = ImageEncoder(
            in_channels, img_enc_output_dim, dropout=img_enc_dropout, **kwargs
        )
        self._measurement_encoder = MeasurementEncoder(
            num_measurements, measurement_enc_output_dim, dropout=measurement_enc_dropout
        )
        self.latent_rep_dim = img_enc_output_dim + measurement_enc_output_dim
        if action_enc_output_dim:
            self.latent_rep_dim += action_enc_output_dim
            self._action_encoder = ContinuousActionEncoder(
                action_dim, action_enc_output_dim, dropout=action_enc_dropout or 0.0
            )
        self._latent_rep = LatentRep(self.latent_rep_dim, hidden_dim)

    def _generate_output_params(
            self, 
            num_intentions: int, 
            input_dim: int,
            output_dim: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        ws = torch.zeros(num_intentions, input_dim, output_dim)
        bs = torch.zeros(num_intentions, 1, output_dim)
        nn.init.kaiming_uniform_(ws, a=math.sqrt(5))
        nn.init.kaiming_uniform_(bs, a=math.sqrt(5))
        _weights = nn.Parameter(ws, requires_grad=True)
        _bias = nn.Parameter(bs, requires_grad=True)
        return _weights, _bias
    
    def forward(*args, **kwargs):
        raise NotImplementedError


class ActorNetwork(_CommonModule):
    def __init__(self, *args, **kwargs):
        super(ActorNetwork, self).__init__(*args, **kwargs)
        self._weights, self._bias =  self._generate_output_params(
            self.num_intentions, self.hidden_dim, self.action_dim
        )

    def forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
        ) -> torch.Tensor:

        img_enc_output = self._image_encoder(cam_obs)
        measurement_enc_output = self._measurement_encoder(measurements)
        output = torch.concat([img_enc_output, measurement_enc_output], dim=1)
        output = self._latent_rep(output)
        output = output.unsqueeze(dim=1)

        intentions = intentions.squeeze(dim=1)
        actor_weights = self._weights[intentions]
        actor_bias = self._bias[intentions]

        output = torch.baddbmm(actor_bias, output, actor_weights).squeeze(dim=1).tanh()
        steer = output[..., :1]
        throttle = (output[..., 1:] + 1) / 2
        action = torch.cat([steer, throttle], dim=-1)
        return action
    

class CriticNetwork(_CommonModule):
    def __init__(self, *args, **kwargs):
        super(CriticNetwork, self).__init__(*args, **kwargs)

        self._weights, self._bias = self._generate_output_params(
            self.num_intentions, self.hidden_dim, 1
        )

    def forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
            actions: torch.FloatTensor
        ) -> torch.Tensor:

        img_enc_output = self._image_encoder(cam_obs)
        measurement_enc_output = self._measurement_encoder(measurements)
        action_enc_output = self._action_encoder(actions)
        output = torch.concat(
            [
                img_enc_output, 
                measurement_enc_output, 
                action_enc_output
            ], dim=1
        )
        output = self._latent_rep(output)
        output = output.unsqueeze(dim=1)

        intentions = intentions.squeeze(dim=1)
        critic_weights = self._weights[intentions]
        critic_bias = self._bias[intentions]
        
        output = torch.baddbmm(critic_bias, output, critic_weights)
        return output


class ActorCriticNetwork(_CommonModule):
    def __init__(self, *args, **kwargs):
        super(ActorCriticNetwork, self).__init__(*args, **kwargs)

        _ad = (self.img_enc_output_dim + self.measurement_enc_output_dim)
        self._actor_latent_rep = LatentRep(_ad, self.hidden_dim)
        self._actor_weights, self._actor_bias = self._generate_output_params(
            self.num_intentions, self.hidden_dim, self.action_dim
        )
        _cd = (self.img_enc_output_dim + self.measurement_enc_output_dim + self.action_enc_output_dim)
        self._critic_latent_rep = LatentRep(_cd, self.hidden_dim)
        self._critic_weights, self._critic_bias = self._generate_output_params(
            self.num_intentions, self.hidden_dim, 1
        )
        del self._latent_rep, self.latent_rep_dim

    def actor_forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
    ) -> torch.Tensor:
        
        img_enc_output = self._image_encoder(cam_obs)
        measurement_enc_output = self._measurement_encoder(measurements)
        output = torch.concat([img_enc_output, measurement_enc_output], dim=1)
        output = self._actor_latent_rep(output)
        output = output.unsqueeze(dim=1)

        intentions = intentions.squeeze(dim=1)
        actor_weights = self._actor_weights[intentions]
        actor_bias = self._actor_bias[intentions]
        output = torch.baddbmm(actor_bias, output, actor_weights).squeeze(dim=1).tanh()
        steer = output[..., :1]
        throttle = (output[..., 1:] + 1) / 2
        action = torch.cat([steer, throttle], dim=-1)
        return action
    
    def critic_forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
            actions: torch.FloatTensor
        ) -> torch.Tensor:

        img_enc_output = self._image_encoder(cam_obs)
        measurement_enc_output = self._measurement_encoder(measurements)
        action_enc_output = self._action_encoder(actions)
        output = torch.concat(
            [
                img_enc_output, 
                measurement_enc_output, 
                action_enc_output
            ], dim=1
        )
        output = self._critic_latent_rep(output)
        output = output.unsqueeze(dim=1)

        intentions = intentions.squeeze(dim=1)
        critic_weights = self._critic_weights[intentions]
        critic_bias = self._critic_bias[intentions]
        
        output = torch.baddbmm(critic_bias, output, critic_weights)
        return output