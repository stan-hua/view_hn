"""
ssl_collate_fn.py

Description: Contains collate functions to augment image batches for
             self-supervised pretraining.
"""

# Non-standard libraries
import lightly
import numpy as np
import torch


################################################################################
#                           CollateFunction Classes                            #
################################################################################
class SimCLRCollateFunction(lightly.data.collate.BaseCollateFunction):
    """
    SimCLRCollateFunction. Used to create paired image augmentations with custom
    batch data.
    """
    def forward(self, batch):
        """
        For each image in the batch, creates two random augmentations on the
        same image and pairs them.

        Parameters
        ----------
            batch: tuple of (torch.Tensor, dict)
                Tuple of image tensors and metadata dict (containing filenames,
                etc.)

        Returns
        -------
            tuple of ((torch.Tensor, torch.Tensor), dict).
                The two tensors consist of corresponding images transformed from
                the original batch.
        """
        batch_size = len(batch)

        # Accumulate lists for keys in metadata dicts
        metadata_accum = {}
        for item in batch:
            metadata = item[1]
            for key, val in metadata.items():
                if key not in metadata_accum:
                    metadata_accum[key] = []
                metadata_accum[key].append(val)

        # Perform random augmentation on each image twice
        X_transformed = [self.transform(batch[i % batch_size][0]).unsqueeze_(0)
            for i in range(2 * batch_size)]

        # Tuple of paired transforms
        X_transformed_paired = (
            torch.cat(X_transformed[:batch_size], 0),
            torch.cat(X_transformed[batch_size:], 0)
        )

        return X_transformed_paired, metadata_accum


class SameLabelCollateFunction(lightly.data.collate.BaseCollateFunction):
    """
    SameLabelCollateFunction. Used to create two augmentations for each images,
    and pair images with the same label.
    """

    def forward(self, batch):
        """
        For each image in the batch, creates two random augmentations on each
        image and pairs same-label images.

        Note
        ----
        Because of two augmentations per image, there is no possibility that
        a label having only 1 image (in the batch) will have no pair. In the
        worst case, it will pair with a random augmentation of itself.

        Parameters
        ----------
            batch: tuple of (torch.Tensor, dict)
                Tuple of image tensors and metadata dict (containing filenames,
                etc.)

        Returns
        -------
            tuple of ((torch.Tensor, torch.Tensor), dict).
                The two tensors with corresponding images that have the same
                label.
        """
        # Accumulate lists for keys in metadata dicts
        metadata_accum = {}
        for item in batch:
            metadata = item[1]
            for key, val in metadata.items():
                if key not in metadata_accum:
                    metadata_accum[key] = []

                if not isinstance(val, str):
                    try:
                        metadata_accum[key].extend(val)
                    except:
                        metadata_accum[key].append(val)
                else:
                    metadata_accum[key].append(val)

        # Convert metadata lists to arrays
        for key, val_list in metadata_accum.items():
            metadata_accum[key] = np.array(val_list)

        # Group by label
        # NOTE: Precondition that label exists
        # NOTE: Duplicate by 2 to account for the two augmentations
        labels = np.concatenate([metadata_accum["label"],
                                 metadata_accum["label"]])
        label_to_indices = {
            label: np.argwhere(labels == label).squeeze() \
                for label in np.unique(labels)
        }

        # Create indices to pair images of the same label
        first_idx = []
        second_idx = []
        for _, indices in label_to_indices.items():
            # Pair randomly selected images
            # Shuffle indices
            n = len(indices)
            chosen_indices = np.random.choice(indices, size=n, replace=False)
            first_idx.extend(chosen_indices[:int(n/2)])
            second_idx.extend(chosen_indices[int(n/2):])

        # Get images
        # If batch size is 1, but contains multiple US images
        if len(batch) == 1 and len(batch[0][0]) > 1:
            imgs = batch[0][0]
        else:
            imgs = [data[0] for data in batch]

        # Perform random augmentation on each image twice
        batch_size = len(imgs)
        X_transformed = [
            self.transform(imgs[i % batch_size]).unsqueeze_(0)
            for i in range(batch_size * 2)]

        # Tuple of paired transforms
        X_transformed_paired = (
            torch.cat([X_transformed[idx] for idx in first_idx], 0),
            torch.cat([X_transformed[idx] for idx in second_idx], 0)
        )

        return X_transformed_paired, metadata_accum
