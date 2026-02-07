# Multimodal Clinical Reasoning for Chest X-Rays

A vision-language model that integrates chest X-ray images with clinical context to generate differential diagnoses with step-by-step reasoning, mimicking radiologist decision-making.

## 🎯 Project Goals

- **Beyond Classification**: Move from simple pathology detection to clinical reasoning
- **Context Integration**: Combine visual findings with patient history
- **Explainability**: Generate human-readable diagnostic reasoning

## 📊 Datasets

- **CheXpert**: 224,316 chest X-rays with 14 pathology labels
- **MIMIC-CXR**: 377,110 chest X-rays with free-text radiology reports

## 🏗️ Architecture
```
Input: Clinical History (text) + Chest X-Ray (image)
       ↓
[Image Encoder] + [Text Encoder]
       ↓
[Cross-Attention Fusion]
       ↓
[Reasoning Decoder (LLM)]
       ↓
Output: Differential Diagnosis with Explanation
```

## 🚀 Quick Start
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/multimodal-cxr-reasoning.git
cd multimodal-cxr-reasoning

# Install dependencies
pip install -r requirements.txt

# Prepare data (after downloading CheXpert)
python scripts/prepare_data.py

# Train model
python scripts/train.py --config configs/base_config.yaml
```

## 📈 Development Timeline

- **Week 1-2**: Data preparation + synthetic notes generation
- **Week 3-4**: Model architecture + baseline training
- **Week 5-6**: MIMIC-CXR integration + fine-tuning
- **Week 7-8**: Evaluation + qualitative analysis
- **Week 9-10**: Paper writing + demo

## 📝 TODO

- [ ] Download CheXpert dataset
- [ ] Generate synthetic clinical notes
- [ ] Implement image encoder (ViT)
- [ ] Implement text encoder (BioClinicalBERT)
- [ ] Implement fusion module
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Request MIMIC-CXR access
- [ ] Radiologist evaluation protocol

## 📚 References

- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/

## 📄 License

MIT License