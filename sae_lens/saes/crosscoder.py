"""
Crosscoder classes for mapping between multiple layers of a language model.

A crosscoder extends the transcoder concept to handle concatenated activations
from multiple input layers mapping to concatenated activations at multiple output layers.
This enables learning cross-layer features that can be causal, acausal, or mixed.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    SAEMetadata,
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
    d_out: int = 768
    d_model: int = 768
    n_input_layers: int = 1
    n_output_layers: int = 1
    
    # Hook names for metadata
    hook_names_in: list[str] | None = None
    hook_names_out: list[str] | None = None
    
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
            "hook_names_in": self.hook_names_in,
            "hook_names_out": self.hook_names_out,
        })
        
        return res
    
    def __post_init__(self):
        if self.apply_b_dec_to_input:
            raise ValueError("apply_b_dec_to_input is not supported for crosscoders")
        
        # Validate dimensions
        expected_d_in = self.d_model * self.n_input_layers
        expected_d_out = self.d_model * self.n_output_layers
        
        if self.d_in != expected_d_in:
            raise ValueError(
                f"d_in ({self.d_in}) must equal d_model ({self.d_model}) × "
                f"n_input_layers ({self.n_input_layers}) = {expected_d_in}"
            )
        if self.d_out != expected_d_out:
            raise ValueError(
                f"d_out ({self.d_out}) must equal d_model ({self.d_model}) × "
                f"n_output_layers ({self.n_output_layers}) = {expected_d_out}"
            )
        
        return super().__post_init__()


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