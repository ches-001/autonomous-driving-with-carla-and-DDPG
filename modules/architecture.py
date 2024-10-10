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
        self.out_features = out_features

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
            num_critics: Optional[int]=None,
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
        self.num_critics = num_critics

        self._image_encoder = ImageEncoder(
            in_channels, img_enc_output_dim, dropout=img_enc_dropout, **kwargs
        )
        self._measurement_encoder = MeasurementEncoder(
            num_measurements, measurement_enc_output_dim, dropout=measurement_enc_dropout
        )
        self.latent_rep_dim = img_enc_output_dim + measurement_enc_output_dim
        if action_enc_output_dim:
            self._action_encoder = ContinuousActionEncoder(
                action_dim, action_enc_output_dim, dropout=action_enc_dropout or 0.0
            )
        self._latent_rep = LatentRep(self.latent_rep_dim, hidden_dim)

    def _generate_output_params(
            self, 
            num_intentions: int, 
            input_dim: int,
            output_dim: int,
            num_layers: int=1
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        ws = torch.zeros(num_layers, num_intentions, input_dim, output_dim)
        bs = torch.zeros(num_layers, num_intentions, 1, output_dim)
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
            self.num_intentions, self.hidden_dim, self.action_dim, num_layers=1
        )

    def forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
        ) -> torch.FloatTensor:

        img_enc_output = self._image_encoder(cam_obs)
        measurement_enc_output = self._measurement_encoder(measurements)
        fmap = self._latent_rep(torch.cat([img_enc_output, measurement_enc_output], dim=1))
        intentions = intentions.squeeze(dim=1)
        output = torch.stack([
            torch.baddbmm(
                self._bias[i, intentions], 
                fmap.unsqueeze(dim=1), 
                self._weights[i, intentions]
            ).squeeze(dim=1).tanh() for i in range(0, self._weights.shape[0])
        ], dim=1).mean(dim=1)

        steer = output[..., :1]
        throttle = (output[..., 1:] + 1) / 2
        action = torch.cat([steer, throttle], dim=-1)
        return action
    

class CriticNetwork(_CommonModule):
    def __init__(self, *args, **kwargs):
        super(CriticNetwork, self).__init__(*args, **kwargs)
        self.num_critics = self.num_critics or 1
        self._weights, self._bias = self._generate_output_params(
            self.num_intentions, self.hidden_dim+self.action_enc_output_dim, 1, num_layers=self.num_critics
        )

    def forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
            actions: torch.FloatTensor,
            reduction: str="mean",
            use_all_critics: bool=True
        ) -> Union[List[torch.FloatTensor], torch.FloatTensor]:
        # setting reduction to amax or amin can lead to overestimation or underestimation
        # bias, usually the latter is more preferable than the former. reduction=mean is a
        # more balanced choice, but just because it is balanced does not mean it is more
        # suitable, since it doesn't properly handle Q overestimation and understimation
        # like amin and amax respectively
        assert reduction in ["mean", "amin", "amax", "none"]
        img_enc_output = self._image_encoder(cam_obs)
        measurement_enc_output = self._measurement_encoder(measurements)
        fmap = self._latent_rep(torch.cat([img_enc_output, measurement_enc_output], dim=1))
        qnet_input = torch.cat([fmap, self._action_encoder(actions)], dim=1)
        intentions = intentions.squeeze(dim=1)
        output = [
            torch.baddbmm(
                self._bias[i, intentions], 
                qnet_input.unsqueeze(dim=1), 
                self._weights[i, intentions]
            ).squeeze(dim=1) for i in range(0, self._weights.shape[0] if use_all_critics else 1)
        ]
        if reduction == "none":
            if len(output) == 1:
                return output[0]
            return output
        return getattr(torch.cat(output, dim=1), reduction)(dim=1, keepdim=True)


class ActorCriticNetwork(_CommonModule):
    def __init__(self, *args, **kwargs):
        super(ActorCriticNetwork, self).__init__(*args, **kwargs)
        self.num_critics = self.num_critics or 1
        self._actor_weights, self._actor_bias = self._generate_output_params(
            self.num_intentions, self.hidden_dim, self.action_dim, num_layers=1
        )
        self._critic_weights, self._critic_bias = self._generate_output_params(
            self.num_intentions, self.hidden_dim+self.action_enc_output_dim, 1, num_layers=self.num_critics
        )

    def actor_forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
    ) -> torch.FloatTensor:
        img_enc_output = self._image_encoder(cam_obs)
        measurement_enc_output = self._measurement_encoder(measurements)
        fmap = self._latent_rep(torch.cat([img_enc_output, measurement_enc_output], dim=1))
        intentions = intentions.squeeze(dim=1)
        output = torch.stack([
            torch.baddbmm(
                self._actor_bias[i, intentions], 
                fmap.unsqueeze(dim=1), 
                self._actor_weights[i, intentions]
            ).squeeze(dim=1).tanh() for i in range(0, self._actor_weights.shape[0])
        ], dim=1).mean(dim=1)
        steer = output[..., :1]
        throttle = (output[..., 1:] + 1) / 2
        action = torch.cat([steer, throttle], dim=-1)
        return action
    
    def critic_forward(
            self, 
            cam_obs: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intentions: torch.LongTensor,
            actions: torch.FloatTensor,
            reduction: str="mean",
            use_all_critics: bool=True
        ) -> Union[List[torch.FloatTensor], torch.FloatTensor]:
        # setting reduction to "amax" or "amin" can lead to overestimation or underestimation
        # bias respectively, usually the latter is more preferable than the former.
        # reduction="mean" is a more balanced choice, but just because it is balanced does not
        # mean it is more suitable or better, since it doesn't properly handle Q-value overestimation
        #  and understimation like amin and amax respectively
        assert reduction in ["mean", "amin", "amax", "none"]
        # since the actor and critic share these two layers, we let only the actor 
        # update function update them, hence we disable the gradient computation for 
        # the critic.
        with torch.set_grad_enabled(False):
            img_enc_output = self._image_encoder(cam_obs)
            measurement_enc_output = self._measurement_encoder(measurements)
            fmap = torch.cat([img_enc_output, measurement_enc_output], dim=1)
            fmap = self._latent_rep(fmap)            
        qnet_input = torch.cat([fmap, self._action_encoder(actions)], dim=1)
        intentions = intentions.squeeze(dim=1)
        output = [
            torch.baddbmm(
                self._critic_bias[i, intentions], 
                qnet_input.unsqueeze(dim=1), 
                self._critic_weights[i, intentions]
            ).squeeze(dim=1) for i in range(0, self._critic_weights.shape[0] if use_all_critics else 1)
        ]
        if reduction == "none":
            if len(output) == 1:
                return output[0]
            return output
        return getattr(torch.cat(output, dim=1), reduction)(dim=1, keepdim=True)