"""
dinov2_extractor.py — DINOv2 ViT-L ReID Feature Extractor

Replaces OSNet-AIN (512-D) with DINOv2 ViT-L (1024-D) for person
re-identification. DINOv2 was pre-trained on 142M images via self-supervised
learning and provides dramatically better cross-domain, cross-viewpoint
features than supervised-only models like OSNet.

Key improvements over OSNet-AIN:
  - Self-supervised pre-training → better domain generalization
  - ViT-L backbone → much larger receptive field than CNN
  - 1024-D features → richer appearance representation
  - Register tokens (dinov2_vitl14_reg) → cleaner feature maps

Supports optional mask-based background removal for cleaner ReID when
segmentation masks are available from the D4SM tracker.

Usage:
    extractor = DINOv2ReIDExtractor(device='cuda')
    features = extractor.extract_features_batch(pil_images, masks=binary_masks)
    # features: torch.Tensor of shape (N, 1024), L2-normalized
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


class DINOv2ReIDExtractor:
    FEATURE_DIM = 1024  # DINOv2 ViT-L output dimension

    def __init__(self, model_name='dinov2_vitl14_reg', device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        print(f"Loading DINOv2 ReID extractor ({model_name}) on {self.device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"DINOv2 ReID ready — {self.FEATURE_DIM}-D features")

    def extract_feature(self, image_input, mask=None):
        """
        Extract a single 1024-D L2-normalized feature vector.

        Args:
            image_input: PIL Image, np.ndarray (BGR), or file path.
            mask: Optional binary mask (H, W) uint8.  Background zeroed before
                  feature extraction when provided.
        Returns:
            torch.Tensor of shape (1, 1024)
        """
        img = self._to_pil(image_input)
        if mask is not None:
            img = self._apply_mask(img, mask)

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
            features = F.normalize(features, p=2, dim=1)
        return features

    def extract_features_batch(self, image_list, masks=None):
        """
        Batch feature extraction with optional per-image masks.

        Args:
            image_list: List of PIL Images, np.ndarrays (BGR), or file paths.
            masks: Optional list of binary masks (H, W) uint8, same length as
                   image_list.  None entries are skipped (no masking).
        Returns:
            torch.Tensor of shape (N, 1024), L2-normalized
        """
        if not image_list:
            return torch.empty(0, self.FEATURE_DIM, device=self.device)

        tensors = []
        for i, img_input in enumerate(image_list):
            img = self._to_pil(img_input)
            if masks is not None and i < len(masks) and masks[i] is not None:
                img = self._apply_mask(img, masks[i])
            tensors.append(self.transform(img))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
        return features

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image_input):
        if isinstance(image_input, str):
            return Image.open(image_input).convert('RGB')
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        return image_input.convert('RGB')

    @staticmethod
    def _apply_mask(pil_image, mask):
        """Zero out background pixels using a binary mask."""
        img_np = np.array(pil_image)
        if mask.shape[:2] != img_np.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        img_np[mask == 0] = 0
        return Image.fromarray(img_np)


# ── quick sanity test ──────────────────────────────────────────────
if __name__ == "__main__":
    ext = DINOv2ReIDExtractor()

    dummy = [Image.new('RGB', (100, 250), c) for c in ['red', 'blue', 'green']]
    feats = ext.extract_features_batch(dummy)
    print(f"Batch shape: {feats.shape}")
    norms = torch.norm(feats, dim=1)
    print(f"L2 norms (should be ~1.0): {norms.tolist()}")
    sim = F.cosine_similarity(feats[0:1], feats[1:2]).item()
    print(f"Cosine sim red↔blue: {sim:.4f}")
