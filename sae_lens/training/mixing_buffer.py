from collections.abc import Iterator

import torch


@torch.no_grad()
def mixing_buffer(
    buffer_size: int,
    batch_size: int,
    activations_loader: Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
) -> Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """
    A generator that maintains a mix of old and new activations for better training.
    It stores half of the activations and mixes them with new ones to create batches.

    Args:
        buffer_size: Total size of the buffer (will store buffer_size/2 activations)
        batch_size: Size of batches to return
        activations_loader: Iterator providing new activations (single tensor or tuple of tensors)

    Yields:
        Batches of activations of shape (batch_size, *activation_dims)
        For crosscoders: yields tuple of (batch_acts_in, batch_acts_out)
    """

    if buffer_size < batch_size:
        raise ValueError("Buffer size must be greater than or equal to batch size")

    storage_buffer_in: torch.Tensor | None = None
    storage_buffer_out: torch.Tensor | None = None
    is_crosscoder = False

    for new_activations in activations_loader:
        # Check if this is crosscoder (tuple) or standard SAE (single tensor)
        if isinstance(new_activations, tuple):
            is_crosscoder = True
            new_acts_in, new_acts_out = new_activations
            
            storage_buffer_in = (
                new_acts_in
                if storage_buffer_in is None
                else torch.cat([storage_buffer_in, new_acts_in], dim=0)
            )
            storage_buffer_out = (
                new_acts_out
                if storage_buffer_out is None
                else torch.cat([storage_buffer_out, new_acts_out], dim=0)
            )
            
            if storage_buffer_in.shape[0] >= buffer_size:
                # Shuffle both buffers with same permutation
                perm = torch.randperm(storage_buffer_in.shape[0])
                storage_buffer_in = storage_buffer_in[perm]
                storage_buffer_out = storage_buffer_out[perm]

                num_serving_batches = max(1, storage_buffer_in.shape[0] // (2 * batch_size))
                serving_cutoff = num_serving_batches * batch_size
                
                serving_buffer_in = storage_buffer_in[:serving_cutoff]
                serving_buffer_out = storage_buffer_out[:serving_cutoff]
                storage_buffer_in = storage_buffer_in[serving_cutoff:]
                storage_buffer_out = storage_buffer_out[serving_cutoff:]

                # Yield batches from the serving_buffers
                for batch_idx in range(num_serving_batches):
                    yield (
                        serving_buffer_in[batch_idx * batch_size : (batch_idx + 1) * batch_size],
                        serving_buffer_out[batch_idx * batch_size : (batch_idx + 1) * batch_size],
                    )
        else:
            # Standard SAE case (single tensor)
            storage_buffer_in = (
                new_activations
                if storage_buffer_in is None
                else torch.cat([storage_buffer_in, new_activations], dim=0)
            )

            if storage_buffer_in.shape[0] >= buffer_size:
                # Shuffle
                storage_buffer_in = storage_buffer_in[torch.randperm(storage_buffer_in.shape[0])]

                num_serving_batches = max(1, storage_buffer_in.shape[0] // (2 * batch_size))
                serving_cutoff = num_serving_batches * batch_size
                serving_buffer = storage_buffer_in[:serving_cutoff]
                storage_buffer_in = storage_buffer_in[serving_cutoff:]

                # Yield batches from the serving_buffer
                for batch_idx in range(num_serving_batches):
                    yield serving_buffer[
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ]

    # If there are any remaining activations, yield them
    if storage_buffer_in is not None:
        remaining_batches = storage_buffer_in.shape[0] // batch_size
        if is_crosscoder and storage_buffer_out is not None:
            for i in range(remaining_batches):
                yield (
                    storage_buffer_in[i * batch_size : (i + 1) * batch_size],
                    storage_buffer_out[i * batch_size : (i + 1) * batch_size],
                )
        else:
            for i in range(remaining_batches):
                yield storage_buffer_in[i * batch_size : (i + 1) * batch_size]