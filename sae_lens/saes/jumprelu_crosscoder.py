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
from sae_lens.saes.jumprelu_sae import JumpReLU, Step, calculate_pre_act_loss
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
class JumpReLUCrosscoderConfig(CrosscoderConfig):
    """Configuration for JumpReLU crosscoder."""

    @classmethod
    def architecture(cls) -> str:
        """Return the architecture name for this config."""
        return "jumprelu_crosscoder"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "JumpReLUCrosscoderConfig":
        """Create a JumpReLUCrosscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)

        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])

        return res


class JumpReLUCrosscoder(Crosscoder):
    """
    A crosscoder with JumpReLU activation function.

    JumpReLU applies a learned threshold to activations: if pre-activation
    is below the threshold, the unit is zeroed out; otherwise, it follows
    the base activation function (ReLU).
    
    This can lead to better sparsity and feature quality by allowing the
    model to learn which features are "real" vs noise.
    """

    cfg: JumpReLUCrosscoderConfig  # type: ignore[assignment]
    threshold: nn.Parameter

    def __init__(self, cfg: JumpReLUCrosscoderConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def initialize_weights(self):
        """Initialize crosscoder weights including threshold parameter."""
        super().initialize_weights()

        # Initialize threshold parameter for JumpReLU
        # One threshold per feature in the sparse bottleneck
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using JumpReLU activation.

        Applies base activation function (ReLU) then masks based on learned threshold.
        During training, the threshold is detached to prevent gradient flow.
        
        Args:
            x: Concatenated input activations [batch, d_model * n_input_layers]
        
        Returns:
            feature_acts: Sparse feature activations [batch, d_sae]
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        
        # Apply base activation function (ReLU)
        feature_acts = self.activation_fn(hidden_pre)

        # Apply JumpReLU threshold
        # During training, use detached threshold to prevent gradient flow
        threshold = self.threshold.detach() if self.training else self.threshold
        jump_relu_mask = (hidden_pre > threshold).to(self.dtype)

        # Apply mask and hook
        return self.hook_sae_acts_post(feature_acts * jump_relu_mask)

    def fold_W_dec_norm(self) -> None:
        """
        Fold the decoder weight norm into the threshold parameter.

        This is important for JumpReLU as the threshold needs to be scaled
        along with the decoder weights to maintain the same effective threshold.
        """
        # Get the decoder weight norms before normalizing
        with torch.no_grad():
            W_dec_norms = self.W_dec.norm(dim=1)

        # Fold the decoder norms as in the parent class
        super().fold_W_dec_norm()

        # Scale the threshold by the decoder weight norms
        with torch.no_grad():
            self.threshold.data = self.threshold.data * W_dec_norms

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "JumpReLUCrosscoder":
        cfg = JumpReLUCrosscoderConfig.from_dict(config_dict)
        return cls(cfg)


@dataclass  
class TrainingJumpReLUCrosscoderConfig(JumpReLUCrosscoderConfig, TrainingSAEConfig):
    """
    Configuration class for training a JumpReLUCrosscoder.

    - jumprelu_init_threshold: initial threshold for the JumpReLU activation
    - jumprelu_bandwidth: bandwidth for the JumpReLU activation
    - jumprelu_sparsity_loss_mode: mode for the sparsity loss, either "step" or "tanh". "step" is Google Deepmind's L0 loss, "tanh" is Anthropic's sparsity loss.
    - l0_coefficient: coefficient for the l0 sparsity loss
    - l0_warm_up_steps: number of warm-up steps for the l0 sparsity loss
    - pre_act_loss_coefficient: coefficient for the pre-activation loss. Set to None to disable. Set to 3e-6 to match Anthropic's setup. Default is None.
    - jumprelu_tanh_scale: scale for the tanh sparsity loss. Only relevant for "tanh" sparsity loss mode. Default is 4.0.
    """

    jumprelu_init_threshold: float = 0.01
    jumprelu_bandwidth: float = 0.05
    # step is Google Deepmind, tanh is Anthropic
    jumprelu_sparsity_loss_mode: Literal["step", "tanh"] = "step"
    l0_coefficient: float = 1.0
    l0_warm_up_steps: int = 0

    # anthropic's auxiliary loss to avoid dead features
    pre_act_loss_coefficient: float | None = None

    # only relevant for tanh sparsity loss mode
    jumprelu_tanh_scale: float = 4.0

    @classmethod
    def architecture(cls) -> str:
        return "jumprelu_crosscoder"
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingJumpReLUCrosscoderConfig":
        """Create a TrainingJumpReLUCrosscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])
        
        return res

class TrainingJumpReLUCrosscoder(JumpReLUCrosscoder, TrainingSAE[TrainingJumpReLUCrosscoderConfig]):
    """
    Training version of JumpReLUCrosscoder.
    
    Similar to the inference-only JumpReLUCrosscoder, but with:
      - A learnable log-threshold parameter (instead of a raw threshold).
      - A specialized auxiliary loss term for sparsity (L0 or tanh).
    """
    
    cfg: TrainingJumpReLUCrosscoderConfig  # type: ignore[assignment]
    log_threshold: nn.Parameter

    def __init__(self, cfg: TrainingJumpReLUCrosscoderConfig):
        # Call Crosscoder.__init__ to properly initialize base parameters
        Crosscoder.__init__(self, cfg)
        self.cfg = cfg
        
        # Store bandwidth for training
        self.bandwidth = cfg.jumprelu_bandwidth
        
        # Initialize log_threshold parameter
        self.log_threshold = nn.Parameter(
            torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            * np.log(cfg.jumprelu_init_threshold)
        )
        
        # Turn off hook_z reshaping for training mode
        self.turn_off_forward_pass_hook_z_reshaping()
        self.mse_loss_fn = lambda sae_out, sae_in: (sae_out - sae_in).pow(2)

    @property
    def threshold(self) -> torch.Tensor:
        """
        Returns the parameterized threshold > 0 for each unit.
        threshold = exp(log_threshold).
        """
        return torch.exp(self.log_threshold)

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
        feature_acts = JumpReLU.apply(hidden_pre, self.threshold, self.bandwidth)
        return feature_acts, hidden_pre  # type: ignore

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate architecture-specific auxiliary loss terms."""

        threshold = self.threshold
        W_dec_norm = self.W_dec.norm(dim=1)
        
        if self.cfg.jumprelu_sparsity_loss_mode == "step":
            l0 = torch.sum(
                Step.apply(hidden_pre, threshold, self.bandwidth),  # type: ignore
                dim=-1,
            )
            l0_loss = (step_input.coefficients["l0"] * l0).mean()
        elif self.cfg.jumprelu_sparsity_loss_mode == "tanh":
            per_item_l0_loss = torch.tanh(
                self.cfg.jumprelu_tanh_scale * feature_acts * W_dec_norm
            ).sum(dim=-1)
            l0_loss = (step_input.coefficients["l0"] * per_item_l0_loss).mean()
        else:
            raise ValueError(
                f"Invalid sparsity loss mode: {self.cfg.jumprelu_sparsity_loss_mode}"
            )
        
        losses = {"l0_loss": l0_loss}

        if self.cfg.pre_act_loss_coefficient is not None:
            losses["pre_act_loss"] = calculate_pre_act_loss(
                self.cfg.pre_act_loss_coefficient,
                threshold,
                hidden_pre,
                step_input.dead_neuron_mask,
                W_dec_norm,
            )
        
        return losses

    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        """Get loss coefficient configuration for JumpReLU training."""
        return {
            "l0": TrainCoefficientConfig(
                value=self.cfg.l0_coefficient,
                warm_up_steps=self.cfg.l0_warm_up_steps,
            ),
        }

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to concatenated multi-layer output.
        
        Overrides parent to match TrainingSAE interface expectations.
        """
        sae_out = feature_acts @ self.W_dec + self.b_dec
        return self.hook_sae_recons(sae_out)

    def fold_W_dec_norm(self) -> None:
        """
        Override to properly handle threshold adjustment with W_dec norms.
        """
        # Save the current threshold before we call the parent method
        current_thresh = self.threshold.clone()

        # Get W_dec norms
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)

        # Call parent implementation to handle W_enc and W_dec adjustment
        # Note: JumpReLUCrosscoder.fold_W_dec_norm calls Crosscoder.fold_W_dec_norm
        # which normalizes W_dec and scales W_enc
        super().fold_W_dec_norm()

        # Scale log_threshold accordingly
        self.log_threshold.data = torch.log(current_thresh * W_dec_norms.squeeze())

    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        """Convert log_threshold to threshold for saving."""
        if "log_threshold" in state_dict:
            threshold = torch.exp(state_dict["log_threshold"]).detach().contiguous()
            del state_dict["log_threshold"]
            state_dict["threshold"] = threshold

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        """Convert threshold to log_threshold for loading."""
        if "threshold" in state_dict:
            threshold = state_dict["threshold"]
            del state_dict["threshold"]
            state_dict["log_threshold"] = torch.log(threshold).detach().contiguous()
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingJumpReLUCrosscoder":
        cfg = TrainingJumpReLUCrosscoderConfig.from_dict(config_dict)
        return cls(cfg)