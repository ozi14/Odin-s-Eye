"""
extractor.py — ReID Feature Extractor for Apple M4 Max (MPS)

Uses OSNet-AIN x1.0 (Adaptive Instance Normalization) with multi-source 
domain generalization weights trained on MSMT17 + Market1501 + DukeMTMC + CUHK03.

This model achieves the best cross-domain generalization in the torchreid Model Zoo:
    - Market1501 (unseen): 73.3 Rank-1 (45.8 mAP)
    - DukeMTMC  (unseen): 65.6 Rank-1 (47.2 mAP)

For WILDTRACK (a completely unseen domain with different camera angles, lighting,
and outdoor setting), the AIN variant's domain-adaptive InstanceNorm layers provide
better feature generalization than standard OSNet trained on a single dataset.

Reference:
    Zhou et al. Learning Generalisable Omni-Scale Representations
    for Person Re-Identification. TPAMI, 2021.
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import gdown

# Import the OSNet-AIN architecture
from .osnet_ain import osnet_ain_x1_0


class PersonReIDExtractor:
    def __init__(self, model_name='osnet_ain_x1_0_ms_d_c', device='mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        print(f"✅ Loading ReID model ({model_name}) on {self.device}...")
        
        # Multi-source domain generalization weights (MSMT17 + DukeMTMC + CUHK03)
        # From torchreid Model Zoo: "Multi-source domain generalization" section
        # Trained with: osnet_ain_x1_0, softmax loss, cosine LR, 50 epochs
        # Architecture: OSNet-AIN x1.0 with Adaptive Instance Normalization
        self.weights_id = '1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB'
        self.weights_file = os.path.join(os.path.dirname(__file__), f"{model_name}.pth.tar")
        
        self._download_weights_if_needed()
        
        # Initialize the OSNet-AIN architecture (same 2.2M params, 512-D output)
        self.model = osnet_ain_x1_0(num_classes=1000, pretrained=False)
        
        # Load the multi-source DG weights
        checkpoint = torch.load(self.weights_file, map_location=self.device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Clean the keys (remove 'module.' if saved via DataParallel)
        model_dict = self.model.state_dict()
        new_state_dict = {}
        matched, skipped = 0, 0
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched += 1
            else:
                skipped += 1
                
        model_dict.update(new_state_dict)
        self.model.load_state_dict(model_dict)
        
        print(f"✅ Loaded {matched} layers, skipped {skipped} (classifier head expected)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ReID preprocessing pipeline (matches torchreid training config)
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _download_weights_if_needed(self):
        if not os.path.exists(self.weights_file):
            print(f"⬇️ Downloading OSNet-AIN multi-source DG weights...")
            url = f'https://drive.google.com/uc?id={self.weights_id}'
            gdown.download(url, self.weights_file, quiet=False)

    def extract_feature(self, image_input):
        """
        Extracts a 512-D L2-normalized feature vector from a cropped person image.
        Args:
            image_input: PIL Image or filepath of the cropped person.
        Returns:
            torch.Tensor: Shape (1, 512)
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')
            
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            # L2 normalization for cosine similarity matching
            features = F.normalize(features, p=2, dim=1)
            
        return features

    def extract_features_batch(self, image_list):
        """
        Batch feature extraction for multiple person crops.
        More efficient than calling extract_feature() in a loop.
        
        Args:
            image_list: List of PIL Images or file paths.
        Returns:
            torch.Tensor: Shape (N, 512) — L2-normalized
        """
        if not image_list:
            return torch.empty(0, 512)
        
        tensors = []
        for img_input in image_list:
            if isinstance(img_input, str):
                img = Image.open(img_input).convert('RGB')
            else:
                img = img_input.convert('RGB')
            tensors.append(self.transform(img))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
        
        return features


# --- Simple Test Block ---
if __name__ == "__main__":
    extractor = PersonReIDExtractor()
    
    # Create dummy images mimicking person crops
    dummy_imgs = [Image.new('RGB', (100, 250), color=c) for c in ['red', 'blue', 'green']]
    
    # Single extraction
    feat = extractor.extract_feature(dummy_imgs[0])
    print(f"✅ Single extraction — Shape: {feat.shape}")
    
    # Batch extraction
    feats = extractor.extract_features_batch(dummy_imgs)
    print(f"✅ Batch extraction  — Shape: {feats.shape}")
    
    # Verify L2 normalization
    norms = torch.norm(feats, dim=1)
    print(f"✅ L2 norms (should be ~1.0): {norms.tolist()}")
    
    # Cross-similarity test
    sim_01 = F.cosine_similarity(feats[0:1], feats[1:2]).item()
    sim_02 = F.cosine_similarity(feats[0:1], feats[2:3]).item()
    print(f"✅ Cosine similarity red↔blue:  {sim_01:.4f}")
    print(f"✅ Cosine similarity red↔green: {sim_02:.4f}")
