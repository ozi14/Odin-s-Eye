"""
dinov2_extractor.py — DINOv2 ViT ReID Feature Extractor

Reused from Odin's Eye v2 with minor adaptation for MOT20 single-camera
tracking context (default model switched to ViT-B for faster inference;
ViT-L available via model_name param).

Extracts L2-normalised appearance feature vectors per person crop for
use as a secondary ReID signal in ByteTrack association.

Usage:
    extractor = DINOv2ReIDExtractor(device='mps')
    features = extractor.extract_features_batch(pil_images)
    # features: torch.Tensor of shape (N, 768), L2-normalized  (ViT-B)
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


_MODEL_DIMS = {
    'dinov2_vitb14':         768,
    'dinov2_vitb14_reg':     768,
    'dinov2_vitl14':        1024,
    'dinov2_vitl14_reg':    1024,
    'dinov2_vits14':         384,
    'dinov2_vits14_reg':     384,
}


class DINOv2ReIDExtractor:

    def __init__(self, model_name: str = 'dinov2_vitb14_reg', device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        self.FEATURE_DIM = _MODEL_DIMS.get(model_name, 768)

        print(f"Loading DINOv2 ReID ({model_name}) on {self.device}…")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name,
                                    verbose=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"DINOv2 ReID ready — {self.FEATURE_DIM}-D features")

    def extract_feature(self, image_input, mask=None):
        img = self._to_pil(image_input)
        if mask is not None:
            img = self._apply_mask(img, mask)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = F.normalize(feat, p=2, dim=1)
        return feat

    def extract_features_batch(self, image_list, masks=None):
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
            feats = self.model(batch)
            feats = F.normalize(feats, p=2, dim=1)
        return feats

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image_input):
        if isinstance(image_input, str):
            return Image.open(image_input).convert('RGB')
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        return image_input.convert('RGB')

    @staticmethod
    def _apply_mask(pil_image, mask):
        img_np = np.array(pil_image)
        if mask.shape[:2] != img_np.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        img_np[mask == 0] = 0
        return Image.fromarray(img_np)


if __name__ == '__main__':
    ext = DINOv2ReIDExtractor()
    dummy = [Image.new('RGB', (64, 128), c) for c in ['red', 'blue', 'green']]
    feats = ext.extract_features_batch(dummy)
    print(f"Batch shape: {feats.shape}")
    print(f"L2 norms: {torch.norm(feats, dim=1).tolist()}")
