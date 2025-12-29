# MHCL: Multi-modal Hierarchical Contrastive Learning for Physiological Signal-Based Vigilance Detection

This repository provides the official PyTorch implementation of the **MHCL** model proposed in the paper:

**“MHCL: Multi-modal Hierarchical Contrastive Learning for Physiological Signal-Based Vigilance Detection”**

MHCL is a multi-modal framework designed for robust vigilance state detection by jointly modeling EEG and EOG signals through hierarchical contrastive learning, cross-modal attention, and domain-adversarial training.

---

## 📌 Overview

The MHCL framework integrates:

- Modality-specific feature embedding for EEG and EOG signals
- Cross-modal attention to enable bidirectional information fusion
- Hierarchical contrastive learning at both local and global levels
- Domain-adversarial learning to enhance cross-subject generalization

The proposed model is lightweight and suitable for real-time vigilance monitoring applications.

---



## 🧠 Model Usage (Minimal Example)

```python
import torch
from model import HCANet

model = HCANet()
eeg = torch.randn(4, 17, 5)
eog = torch.randn(4, 36)

outputs = model(eeg, eog)
logits = outputs[0]  # EEG-only prediction
```

