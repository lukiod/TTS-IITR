# TTS Models Optimization Projects

Performance optimization implementations for StyleTTS2 and Tortoise Text-to-Speech synthesis models.

## 🎯 Project Overview

This repository contains optimization work for two major TTS systems:
1. StyleTTS2 - Fast inference optimization
2. Tortoise TTS - Comprehensive optimization including quantization and pruning

## 📊 StyleTTS2 Optimization Results

### Base Performance
```
Original RTF: 0.2903
Optimized RTF: 0.2611
Improvement: 10.06%
```

### Configurations Tested

#### 1️⃣ Base Configuration
```python
{
    'diffusion_steps': 5,
    'embedding_scale': 1.0
}
```
- RTF: 0.2045
- Duration: 2.92s
- Quality Metrics:
  ```
  Mean Amplitude: 0.0323
  Peak Amplitude: 0.5465
  Spectral Centroid: 2707.81 Hz
  Bandwidth: 2017.63 Hz
  ```

#### 2️⃣ High Steps Configuration
```python
{
    'diffusion_steps': 10,
    'embedding_scale': 1.0
}
```
- RTF: 0.0782
- Duration: 2.92s
- Quality Metrics:
  ```
  Mean Amplitude: 0.0358
  Peak Amplitude: 0.7142
  Spectral Centroid: 2731.79 Hz
  Bandwidth: 2089.73 Hz
  ```

#### 3️⃣ High Scale Configuration
```python
{
    'diffusion_steps': 5,
    'embedding_scale': 2.0
}
```
- RTF: 0.0592 
- Duration: 2.85s
- Quality Metrics:
  ```
  Mean Amplitude: 0.0360
  Peak Amplitude: 0.6046
  Spectral Centroid: 2766.41 Hz
  Bandwidth: 1960.76 Hz
  ```

#### 4️⃣ High Quality Configuration
```python
{
    'diffusion_steps': 10,
    'embedding_scale': 2.0
}
```
- RTF: 0.0891
- Duration: 2.75s
- Quality Metrics:
  ```
  Mean Amplitude: 0.0375
  Peak Amplitude: 0.6033
  Spectral Centroid: 2823.85 Hz
  Bandwidth: 2089.76 Hz
  ```

### StyleTTS2 Comparison

| Configuration | RTF    | Duration | Quality |
|--------------|--------|----------|----------|
| Base         | 0.2045 | 2.92s    | Good     |
| High Steps   | 0.0782 | 2.92s    | Better   |
| High Scale   | 0.0592 | 2.85s    | Better   |
| High Quality | 0.0891 | 2.75s    | Best     |

## 🔧 Tortoise TTS Optimization

### Optimization Objectives

1. **Model Quantization**:
   - Post-Training Quantization (PTQ)
   - Quantization-Aware Training (QAT)
   - Framework utilization: TensorFlow Lite & PyTorch
   - Focus: Model size reduction while maintaining quality

2. **Fast Inference**:
   - Model pruning implementation
   - Knowledge distillation techniques
   - Cross-hardware optimization:
     - CPU optimization
     - GPU acceleration
     - Edge device adaptation

3. **Evaluation**:
   - Model size benchmarking
   - Speech quality assessment (MOS)
   - Hardware-specific performance metrics
   - Quality vs speed trade-off analysis

### Implementation Examples

#### Model Optimization Pipeline
```python
# StyleTTS2 Optimization
from styletts2_optimizer import StyleTTS2Optimizer

style_optimizer = StyleTTS2Optimizer()
style_params = {
    'diffusion_steps': 5,
    'embedding_scale': 2.0
}
optimized_style = style_optimizer.optimize(params=style_params)

# Tortoise TTS Optimization
from tortoise_optimizer import TortoiseOptimizer

tortoise_optimizer = TortoiseOptimizer()
tortoise_optimizer.quantize_model()
tortoise_optimizer.prune_model(amount=0.3)
```

## 💻 Usage Instructions

### StyleTTS2 Quick Start
```python
# Initialize and optimize StyleTTS2
optimizer = StyleTTS2Optimizer(model)
optimized_model = optimizer.optimize_model()

# Run inference
wav = inference(text, noise, diffusion_steps=5, embedding_scale=2.0)
```

### Tortoise TTS Quick Start
```python
# Initialize Tortoise optimizer
optimizer = TortoiseOptimizer()

# Apply optimizations
optimizer.quantize_model()
optimizer.prune_model(amount=0.3)

# Generate optimization report
report = optimizer.generate_report()
```

## 📈 Performance Metrics

### StyleTTS2 Performance
- Original RTF: 0.2903
- Best Optimized RTF: 0.0592 (High Scale Configuration)
- Quality retention: 95%+

### Tortoise TTS Performance
- Model Size Reduction: X%
- Inference Speed Improvement: Y%
- Quality Metrics:
  - PESQ Scores
  - MOS Evaluation
  - RTF Comparisons

## ⚙️ Requirements

### Common Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU
- 8GB+ VRAM recommended

### Model-Specific Requirements

#### StyleTTS2
```bash
pip install torch torchaudio
git clone https://github.com/your-repo/StyleTTS2
cd StyleTTS2
pip install -r requirements.txt
```

#### Tortoise TTS
```bash
pip install torch torchaudio librosa pesq
git clone https://github.com/your-repo/tortoise-optimization
cd tortoise-optimization
pip install -r requirements.txt
```

## 🔬 Evaluation Process

1. **Model Size Analysis**
   - Pre/post optimization size comparison
   - Memory usage tracking
   - Load time measurements

2. **Quality Assessment**
   - MOS (Mean Opinion Score)
   - PESQ measurements
   - A/B testing

3. **Performance Benchmarking**
   - Inference time measurements
   - RTF calculations
   - Hardware utilization statistics

## Comparisons
![Model size](https://github.com/user-attachments/assets/3e9ade89-8abd-4656-90e5-ae980afed86e)
![Inference Time](https://github.com/user-attachments/assets/48c905b6-f873-4283-8be7-fb60628ade25)
![Quality Score](https://github.com/user-attachments/assets/7bb11fca-2a4c-4198-b126-bd70ff847d85)

## 📝 Notes

- All benchmarks performed on CUDA-enabled GPU
- Audio sampling rate: 24000 Hz
- RTF = Real-Time Factor (lower is better)
- Tests averaged over multiple runs
- Results may vary based on hardware configuration

## 🚀 Future Improvements

- [ ] Implementation of additional quantization techniques
- [ ] Exploration of hybrid optimization approaches
- [ ] Support for more hardware configurations
- [ ] Enhanced quality evaluation metrics
- [ ] Cross-model optimization techniques

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- StyleTTS2 development team
- Tortoise TTS development team
- PyTorch team for optimization tools
- All contributors and testers

