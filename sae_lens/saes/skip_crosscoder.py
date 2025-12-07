"""
Crosscoder classes for mapping between multiple layers of a language model.

A crosscoder extends the transcoder concept to handle concatenated activations
from multiple input layers mapping to concatenated activations at multiple output layers.
This enables learning cross-layer features that can be causal, acausal, or mixed.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from sae_lens.saes.crosscoder import Crosscoder, CrosscoderConfig
# from sae_lens.saes.jumprelu_sae import JumpReLU, Step, calculate_pre_act_loss
from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    SAEMetadata,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainCoefficientConfig,
)
from sae_lens.util import filter_valid_dataclass_fields

@dataclass
class SkipCrosscoderConfig(CrosscoderConfig):
    """Configuration for crosscoder with skip connection."""
    
    @classmethod
    def architecture(cls) -> str:
        """Return the architecture name for this config."""
        return "skip_crosscoder"
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SkipCrosscoderConfig":
        """Create a SkipCrosscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])
        
        return res


class SkipCrosscoder(Crosscoder):
    """
    A crosscoder with a learnable skip connection.

    Implements: f(x) = W_dec @ relu(W_enc @ x + b_enc) + W_skip @ x + b_dec
    
    The skip connection W_skip maps directly from concatenated input layers
    to concatenated output layers, allowing the model to learn both sparse
    features and direct linear transformations.
    """

    cfg: SkipCrosscoderConfig  # type: ignore[assignment]
    W_skip: nn.Parameter

    def __init__(self, cfg: SkipCrosscoderConfig):
        super().__init__(cfg)
        self.cfg = cfg

        # Initialize skip connection matrix
        # Shape: [d_out, d_in] to map from concatenated input to concatenated output
        self.W_skip = nn.Parameter(
            torch.zeros(self.cfg.d_out, self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for skip crosscoder.

        Args:
            x: Concatenated input activations [batch, d_model * n_input_layers]

        Returns:
            sae_out: Concatenated output activations [batch, d_model * n_output_layers]
        """
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # Add skip connection: x @ W_skip.T
        # x has shape [batch, d_in], W_skip has shape [d_out, d_in]
        skip_out = x @ self.W_skip.T.to(x.device)
        return sae_out + skip_out

    def forward_with_activations(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and feature activations.

        Args:
            x: Concatenated input activations [batch, d_model * n_input_layers]

        Returns:
            sae_out: Concatenated output activations [batch, d_model * n_output_layers]
            feature_acts: Sparse feature activations [batch, d_sae]
        """
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # Add skip connection
        skip_out = x @ self.W_skip.T.to(x.device)
        sae_out = sae_out + skip_out

        return sae_out, feature_acts

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SkipCrosscoder":
        cfg = SkipCrosscoderConfig.from_dict(config_dict)
        return cls(cfg)


@dataclass
class TrainingSkipCrosscoderConfig(SkipCrosscoderConfig, TrainingSAEConfig):
    """Configuration for training a skip crosscoder."""
    
    @classmethod
    def architecture(cls) -> str:
        return "skip_crosscoder"
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSkipCrosscoderConfig":
        """Create a TrainingSkipCrosscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])
        
        return res


class TrainingSkipCrosscoder(SkipCrosscoder, TrainingSAE[TrainingSkipCrosscoderConfig]):
    """
    Training version of SkipCrosscoder.
    
    Extends SkipCrosscoder with training-specific functionality including
    encode_with_hidden_pre for accessing pre-activations during training.
    """
    
    cfg: TrainingSkipCrosscoderConfig  # type: ignore[assignment]
    
    def __init__(self, cfg: TrainingSkipCrosscoderConfig):
        # Call Crosscoder.__init__ to avoid initializing W_skip twice
        Crosscoder.__init__(self, cfg)
        self.cfg = cfg
        
        # Initialize skip connection matrix
        self.W_skip = nn.Parameter(
            torch.zeros(self.cfg.d_out, self.cfg.d_in, dtype=self.dtype, device=self.device)
        )
        
        # Turn off hook_z reshaping for training mode
        self.turn_off_forward_pass_hook_z_reshaping()
        self.mse_loss_fn = lambda sae_out, sae_in: (sae_out - sae_in).pow(2)
    
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode with access to pre-activation values for training.
        
        Args:
            x: Concatenated input activations [batch, d_model * n_input_layers]
        
        Returns:
            feature_acts: Sparse feature activations [batch, d_sae]
            hidden_pre: Pre-activation values [batch, d_sae]
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        feature_acts = self.activation_fn(hidden_pre)
        return feature_acts, hidden_pre
    
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate auxiliary loss terms for training.
        
        For skip crosscoder, no auxiliary losses are needed.
        """
        return {}
    
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        """
        Get loss coefficient configuration.
        
        For skip crosscoder, no additional coefficients are needed.
        """
        return {}
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to concatenated multi-layer output.
        
        Note: This does NOT include the skip connection during training.
        The skip connection is added in training_forward_pass.
        """
        sae_out = feature_acts @ self.W_dec + self.b_dec
        return self.hook_sae_recons(sae_out)
    
    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> Any:  # Returns TrainStepOutput
        """
        Forward pass during training.
        
        Overrides parent to properly handle skip connection in loss calculation.
        """
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out_no_skip = self.decode(feature_acts)
        
        # Add skip connection for final output
        skip_out = step_input.sae_in @ self.W_skip.T.to(step_input.sae_in.device)
        sae_out = sae_out_no_skip + skip_out
        
        # Calculate MSE loss with skip connection included
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()
        
        # Calculate auxiliary losses (if any)
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )
        
        # Total loss
        total_loss = mse_loss
        losses = {"mse_loss": mse_loss}
        
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        
        # Import here to avoid circular dependency
        from sae_lens.saes.sae import TrainStepOutput
        
        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSkipCrosscoder":
        cfg = TrainingSkipCrosscoderConfig.from_dict(config_dict)
        return cls(cfg)
