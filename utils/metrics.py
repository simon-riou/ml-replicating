import torch
import numpy as np
from scipy import linalg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


# TODO: Redo these functions alone + UNDERSTAND IT
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Fréchet Distance between two multivariate Gaussians.

    The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    Args:
        mu1: Mean of first distribution (numpy array)
        sigma1: Covariance matrix of first distribution (numpy array)
        mu2: Mean of second distribution (numpy array)
        sigma2: Covariance matrix of second distribution (numpy array)
        eps: Small value to add to diagonal for numerical stability

    Returns:
        Fréchet distance as a float
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of covariance estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, model, batch_size=50, device='cuda'):
    """
    Calculate mean and covariance statistics of activations from InceptionV3.

    Args:
        images: Tensor of images (N, C, H, W) in range [0, 1]
        model: InceptionV3 model for feature extraction
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        Tuple of (mean, covariance) as numpy arrays
    """
    model.eval()

    activations = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)

            # Resize to 299x299 for InceptionV3
            if batch.shape[2] != 299 or batch.shape[3] != 299:
                batch = torch.nn.functional.interpolate(
                    batch, size=(299, 299), mode='bilinear', align_corners=False
                )

            # Get features
            feat = model(batch)
            activations.append(feat.cpu().numpy())

    activations = np.concatenate(activations, axis=0)

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_fid(real_images, generated_images, device='cuda', batch_size=50):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.

    FID is a metric for evaluating the quality of generated images by comparing
    the statistics of features extracted from an InceptionV3 network.

    Lower FID values indicate better quality and diversity of generated images.

    Args:
        real_images: Tensor of real images (N, C, H, W) in range [0, 1]
        generated_images: Tensor of generated images (N, C, H, W) in range [0, 1]
        device: Device to use for computation
        batch_size: Batch size for processing images

    Returns:
        FID score as a float

    Example:
        >>> real = torch.rand(1000, 3, 32, 32)
        >>> generated = torch.rand(1000, 3, 32, 32)
        >>> fid_score = calculate_fid(real, generated)
        >>> print(f"FID: {fid_score:.2f}")
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        raise ImportError("torchvision is required for FID calculation")

    # Load InceptionV3 model
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception = inception.to(device)
    inception.eval()

    # Calculate statistics for real and generated images
    mu_real, sigma_real = calculate_activation_statistics(
        real_images, inception, batch_size, device
    )
    mu_gen, sigma_gen = calculate_activation_statistics(
        generated_images, inception, batch_size, device
    )

    # Calculate FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid_value