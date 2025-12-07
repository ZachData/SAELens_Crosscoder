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

# from sae_lens.saes.jumprelu_sae import JumpReLU, Step, calculate_pre_act_loss
from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    SAEMetadata,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
    TrainCoefficientConfig,
)
from sae_lens.util import filter_valid_dataclass_fields

@dataclass
class CrosscoderConfig(SAEConfig):
    """
    Configuration for a crosscoder that maps between multiple layers.
    
    A crosscoder takes concatenated activations from n input layers and maps
    them to concatenated activations at m output layers through a sparse bottleneck.
    
    Args:
        d_in: Total input dimension (d_model * n_input_layers)
        d_out: Total output dimension (d_model * n_output_layers)
        d_sae: Sparse bottleneck dimension
        d_model: Base dimension per layer (e.g., 768 for GPT-2)
        n_input_layers: Number of input layers being concatenated
        n_output_layers: Number of output layers being concatenated
        hook_names_in: List of input hook names (for metadata/validation)
        hook_names_out: List of output hook names (for metadata/validation)
    """
    # Output dimension fields
    d_out: int | None = None
    d_model: int | None = None
    n_input_layers: int = 1
    n_output_layers: int = 1
    apply_b_dec_to_input: bool = False
    
    # Hook names for metadata
    from dataclasses import field

    hook_names_in: str = ""
    hook_names_out: str = ""

    @classmethod
    def architecture(cls) -> str:
        """Return the architecture name for this config."""
        return "crosscoder"
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CrosscoderConfig":
        """Create a CrosscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])
        
        return res
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, including parent fields."""
        res = super().to_dict()
        
        res.update({
            "d_out": self.d_out,
            "d_model": self.d_model,
            "n_input_layers": self.n_input_layers,
            "n_output_layers": self.n_output_layers,
            "hook_names_in": ",".join(self.hook_names_in) if isinstance(self.hook_names_in, list) else self.hook_names_in,
            "hook_names_out": ",".join(self.hook_names_out) if isinstance(self.hook_names_out, list) else self.hook_names_out,
        })
        
        return res
    
    def __post_init__(self):
        # Parse comma-separated hook names into lists
        # Handle both str and list[str] inputs from parser
        if isinstance(self.hook_names_in, str):
            self.hook_names_in = [h.strip() for h in self.hook_names_in.split(",") if h.strip()]
        elif isinstance(self.hook_names_in, list) and len(self.hook_names_in) == 1 and "," in self.hook_names_in[0]:
            # Parser gave us ['hook1,hook2'] - split it
            self.hook_names_in = [h.strip() for h in self.hook_names_in[0].split(",") if h.strip()]
        elif not self.hook_names_in:
            self.hook_names_in = []
            
        if isinstance(self.hook_names_out, str):
            self.hook_names_out = [h.strip() for h in self.hook_names_out.split(",") if h.strip()]
        elif isinstance(self.hook_names_out, list) and len(self.hook_names_out) == 1 and "," in self.hook_names_out[0]:
            self.hook_names_out = [h.strip() for h in self.hook_names_out[0].split(",") if h.strip()]
        elif not self.hook_names_out:
            self.hook_names_out = []
        
        if self.apply_b_dec_to_input:
            raise ValueError("apply_b_dec_to_input is not supported for crosscoders")
        
        # Skip dimensional validation if any required field is None (parser introspection mode)
        if None in (self.d_in, self.d_sae, self.d_out, self.d_model, self.n_input_layers, self.n_output_layers):
            return
        
        # Validate dimensions only when we have real values
        expected_d_out = self.d_model * self.n_output_layers
        
        if self.d_out != expected_d_out:
            raise ValueError(
                f"d_out ({self.d_out}) must equal d_model ({self.d_model}) × "
                f"n_output_layers ({self.n_output_layers}) = {expected_d_out}"
            )
        
        # Call parent validation if it exists (handles multiple inheritance MRO)
        try:
            super().__post_init__()
        except AttributeError:
            pass

class Crosscoder(SAE[CrosscoderConfig]):
    """
    A crosscoder maps concatenated activations from multiple input layers
    to concatenated activations at multiple output layers through a sparse bottleneck.
    
    Architecture:
        Input: [batch, d_model * n_input_layers]
        Encode: [batch, d_sae] (sparse features)
        Decode: [batch, d_model * n_output_layers]
    
    This enables learning cross-layer features that can be:
    - Causal: output layers are later than input layers
    - Acausal: output layers are earlier than input layers
    - Mixed: output layers span both earlier and later positions
    """

    cfg: CrosscoderConfig
    W_enc: nn.Parameter
    b_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter

    def __init__(self, cfg: CrosscoderConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def initialize_weights(self):
        """Initialize crosscoder weights with proper dimensions."""
        # Initialize b_dec with output dimension
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_out, dtype=self.dtype, device=self.device)
        )

        # Initialize W_dec with shape [d_sae, d_out]
        w_dec_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_out, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_data)
        self.W_dec = nn.Parameter(w_dec_data)

        # Initialize W_enc with shape [d_in, d_sae]
        w_enc_data = torch.empty(
            self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_enc_data)
        self.W_enc = nn.Parameter(w_enc_data)

        # Initialize b_enc
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def process_sae_in(self, sae_in: torch.Tensor) -> torch.Tensor:
        """
        Process input without applying decoder bias.
        
        Overrides the parent method to skip the bias subtraction since b_dec
        has dimension d_out which doesn't match the input dimension d_in.
        """
        sae_in = sae_in.to(self.dtype)
        sae_in = self.hook_sae_input(sae_in)
        return self.run_time_activation_norm_fn_in(sae_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode concatenated multi-layer input into sparse feature space.
        
        Args:
            x: Concatenated input activations [batch, d_model * n_input_layers]
        
        Returns:
            feature_acts: Sparse feature activations [batch, d_sae]
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to concatenated multi-layer output.
        
        Args:
            feature_acts: Sparse feature activations [batch, d_sae]
        
        Returns:
            sae_out: Concatenated output activations [batch, d_model * n_output_layers]
        """
        sae_out = feature_acts @ self.W_dec + self.b_dec
        return self.hook_sae_recons(sae_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for crosscoder.

        Args:
            x: Concatenated input activations [batch, d_model * n_input_layers]

        Returns:
            sae_out: Concatenated output activations [batch, d_model * n_output_layers]
        """
        feature_acts = self.encode(x)
        return self.decode(feature_acts)

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
        return sae_out, feature_acts

    @property
    def d_out(self) -> int:
        """Output dimension of the crosscoder."""
        return self.cfg.d_out

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Crosscoder":
        cfg = CrosscoderConfig.from_dict(config_dict)
        return cls(cfg)

@dataclass
class TrainingCrosscoderConfig(CrosscoderConfig, TrainingSAEConfig):
    """Configuration for training a crosscoder."""
    
    @classmethod
    def architecture(cls) -> str:
        return "crosscoder"
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingCrosscoderConfig":
        """Create a TrainingCrosscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])
        
        return res

class TrainingCrosscoder(Crosscoder, TrainingSAE[TrainingCrosscoderConfig]):
    """
    Training version of Crosscoder.
    
    Extends the base Crosscoder with training-specific functionality including
    encode_with_hidden_pre for accessing pre-activations during training.
    """
    
    def __init__(self, cfg: TrainingCrosscoderConfig):
        super().__init__(cfg)
    
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
        
        For the basic crosscoder, no auxiliary losses are needed.
        Subclasses can override this to add architecture-specific losses.
        """
        return {}
    
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        """
        Get loss coefficient configuration.
        
        For the basic crosscoder, no additional coefficients are needed.
        """
        return {}
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to concatenated multi-layer output.
        
        Overrides parent to match TrainingSAE interface expectations.
        """
        sae_out = feature_acts @ self.W_dec + self.b_dec
        return self.hook_sae_recons(sae_out)
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingCrosscoder":
        cfg = TrainingCrosscoderConfig.from_dict(config_dict)
        return cls(cfg)
    
    
    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training for crosscoder."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        # For crosscoders, reconstruct the target (output layer activations)
        # not the input (input layer activations)
        target = step_input.target if step_input.target is not None else step_input.sae_in
        
        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, target)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        # Total loss is MSE plus all auxiliary losses
        total_loss = mse_loss
        losses = {"mse_loss": mse_loss}

        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )