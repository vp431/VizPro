# Single-Page Transformers Visualization Tool

An interactive web application for visualizing and analyzing transformer models with advanced explainability features. Built with Dash and supporting multiple NLP tasks including sentiment analysis, named entity recognition, and question answering.

## 🚀 Features

### Core Visualization Capabilities
- **Attention Visualization**: Interactive heatmaps for attention patterns across layers and heads
- **Token Embeddings**: Visualize token representations in semantic space
- **Attention Entropy**: Analyze attention distribution patterns
- **Logit Matrix Heatmaps**: Understand model decision boundaries

### Task-Specific Analysis

#### 🎭 Sentiment Analysis
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Error Analysis**: Comprehensive failure pattern identification
- **Counterfactual Testing**: Understand model behavior through perturbations
- **Similarity Analysis**: Find patterns in model errors using clustering

#### 🏷️ Named Entity Recognition (NER)
- **Entity Visualization**: Interactive entity highlighting and classification
- **Attention Analysis**: See how models focus on entity boundaries
- **Performance Metrics**: Detailed evaluation on CoNLL2003 dataset

#### ❓ Question Answering
- **Knowledge Assessment**: Evaluate model's factual knowledge
- **Counterfactual Flow**: Advanced perturbation testing for QA models
- **Answer Visualization**: Understand how models extract answers
- **Model Comparison**: Compare performance across different architectures

### Advanced Analysis Tools
- **Error Pattern Recognition**: Automatic categorization of model failures
- **t-SNE Clustering**: Visual exploration of error similarity
- **TF-IDF Analysis**: Term importance analysis for error cases
- **Interactive Dashboards**: Real-time model performance monitoring

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- At least 8GB RAM (16GB recommended for large models)
- CUDA-compatible GPU (optional, for faster inference)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/vp431/VizPro.git
   cd VizPro
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models and datasets**
   ```bash
   # Download required models (this may take some time)
   python download_models.py
   
   # Download sample datasets
   python download_datasets.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   Navigate to `http://localhost:8050`

## 📁 Project Structure

```
VizPro/
├── app.py                          # Main application entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── download_models.py              # Model download utility
├── download_datasets.py            # Dataset download utility
├── assets/                         # CSS and static files
│   ├── style.css
│   ├── error_analysis_styles.css
│   ├── qa_styles.css
│   └── ...
├── components/                     # Reusable UI components
│   ├── visualizations.py          # Visualization functions
│   ├── qa_bertviz.py              # QA-specific visualizations
│   └── counterfactual_visualizations.py
├── models/                         # Model management and APIs
│   ├── api.py                     # Main model API
│   ├── sentiment.py               # Sentiment analysis models
│   ├── ner.py                     # NER models
│   ├── qa.py                      # QA models
│   ├── error_analysis.py          # Error analysis tools
│   └── similarity_analysis.py     # Similarity analysis tools
├── pages/                         # Application pages/modules
│   ├── sentiment_lime.py          # LIME analysis page
│   ├── error_analysis.py          # Error analysis dashboard
│   ├── qa_counterfactual_flow.py  # QA counterfactual testing
│   └── ...
├── utils/                         # Utility functions
│   ├── dataset_scanner.py         # Dataset management
│   └── model_scanner.py           # Model discovery and validation
├── LocalModels/                   # Downloaded models (created automatically)
└── LocalDatasets/                 # Downloaded datasets (created automatically)
```

## 🎯 Usage

### Getting Started

1. **Select a Task**: Choose from Sentiment Analysis, NER, or QA
2. **Choose Analysis Level**: 
   - **Sentence-level**: Analyze individual inputs
   - **Model-level**: Comprehensive model evaluation
3. **Input Text**: Enter your text or select from example datasets
4. **Explore Features**: Click feature buttons to access different analysis tools

### Key Features Guide

#### Sentiment Analysis
```python
# Example usage in the interface:
# 1. Select "Sentiment" task
# 2. Enter: "This movie was terrible, but the acting was great!"
# 3. Click "LIME" to see word importance
# 4. Use "Error Analysis" to understand model failures
```

#### Error Analysis Workflow
1. Navigate to "Error Analysis" in model-level features
2. Set sample size (100-500) and confidence threshold
3. Review categorized error patterns:
   - Negation handling errors
   - Intensity modifier errors
   - Context complexity errors
   - Sarcasm/irony detection errors
4. Use similarity analysis to find error clusters
5. Apply counterfactual testing to understand failure causes

#### QA Counterfactual Testing
1. Select "QA" task and "Counterfactual Flow"
2. Input context and question
3. Explore automatic perturbations:
   - Entity replacement
   - Context modification
   - Question paraphrasing
4. Analyze answer consistency and model robustness

## 🔧 Configuration

### Model Configuration
Edit `config.py` to customize:
- Default models for each task
- Local storage paths
- Analysis parameters

### UI Customization
Modify CSS files in `assets/` to customize:
- Color schemes
- Layout styles
- Interactive elements

## 📊 Supported Models

### Pre-configured Models
- **Sentiment**: DistilBERT-SST2, TinyBERT
- **NER**: BERT-base-NER, DistilBERT-NER
- **QA**: DistilBERT-SQuAD, BERT-base-SQuAD

### Adding Custom Models
1. Add model configuration to `config.py`
2. Implement model wrapper in appropriate `models/` file
3. Update UI components if needed

## 📈 Datasets

### Supported Datasets
- **SST-2**: Stanford Sentiment Treebank
- **CoNLL-2003**: Named Entity Recognition
- **SQuAD v1.1**: Question Answering
- **IMDb**: Movie Review Sentiment

### Adding Custom Datasets
1. Add dataset configuration to `config.py`
2. Implement loader in `utils/dataset_scanner.py`
3. Update UI to display new dataset options

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 .
black .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For transformer models and datasets
- **LIME**: For local interpretability methods
- **BertViz**: For attention visualization techniques
- **Plotly/Dash**: For interactive web interface
- **scikit-learn**: For similarity analysis and clustering

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/vp431/VizPro/issues)
- **Documentation**: See `/docs` folder for detailed guides
- **Examples**: Check `/examples` for usage examples

## 🔗 Related Projects

- [Azimuth](https://github.com/ServiceNow/azimuth): Model performance monitoring
- [Errudite](https://github.com/uwdata/errudite): Error analysis framework
- [BertViz](https://github.com/jessevig/bertviz): Attention visualization
- [LIME](https://github.com/marcotcr/lime): Local interpretability

---

**Built with ❤️ at ITT Delhi for the community**
