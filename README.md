# DocBERT - Improved Document Classification with BERT

This repository contains an improved implementation of BERT for document classification, combining techniques from [jesse-tong/docbert](https://github.com/jesse-tong/docbert) and [castorini/hedwig](https://github.com/castorini/hedwig).

## Key Improvements

1. **Advanced Regularization Techniques**:
   - Dropout in multiple layers
   - Layer normalization
   - Gradient clipping
   - Weight decay optimization

2. **Training Stability Enhancements**:
   - Learning rate scheduling with ReduceLROnPlateau
   - Gradient accumulation for effective larger batch sizes
   - Label smoothing to improve generalization
   - Early stopping based on validation F1 score

3. **Architectural Changes**:
   - Better BERT pooling strategies
   - More robust tokenization with attention masks
   - Configurable hyperparameters for different document types

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/docbert-improved.git
cd docbert-improved

# Install dependencies
pip install -r requirements.txt