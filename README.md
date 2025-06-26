# TinyBERT Attention Visualization Tool

This interactive web application visualizes attention patterns in TinyBERT models and helps analyze model performance on sentiment analysis tasks. It provides tools for understanding model behavior, identifying error patterns, and improving model interpretability.

## Features

### 1. Attention Visualization
- Visualize attention maps for any input sentence
- Explore different attention heads and layers
- See how attention flows between tokens in the input

### 2. Sentiment Analysis
- Analyze sentiment predictions on example sentences
- Understand which parts of a sentence influence the prediction
- Compare correct and incorrect predictions

### 3. Named Entity Recognition
- Visualize how the model identifies entities in text
- Compare predicted entities with ground truth

### 4. Explainable AI with LIME
- Understand individual predictions using LIME (Local Interpretable Model-agnostic Explanations)
- See which words contribute most to sentiment predictions
- Analyze feature importance for specific examples

### 5. Enhanced Model Error Analysis
- **Identify patterns in model failures on the SST-2 dataset**
- **Advanced error categorization with smart tagging system:**
  - Negation handling errors (not, n't, never)
  - Intensity modifier errors (very, extremely, really)
  - Context complexity errors (long sentences)
  - Comparison/contrast errors (but, however, although)
  - Sarcasm/irony detection errors
  - Ambiguity errors (words with context-dependent sentiment)
  - Conditional statement errors (if, would, could)
- **Counterfactual testing to understand error causes:**
  - Automatically generate perturbations of error examples
  - Test how small changes affect model predictions
  - Identify minimal changes that fix misclassifications
- **Similarity-based error exploration:**
  - TF-IDF based similarity search to find patterns in error cases
  - Interactive t-SNE visualization of error neighborhoods
  - Clustering of error cases to identify common failure modes
  - Nearest-neighbor analysis to find similar error patterns
- **Comprehensive analysis visualization:**
  - Interactive performance metrics and error distribution
  - Categorized error examples with descriptions
  - Top 15 most frequent words in error cases
  - LIME analysis of specific error cases

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
python app.py
```
4. Open your browser and navigate to http://localhost:8000

## Azure Deployment Instructions

To deploy this application on Azure App Service:

1. Create an Azure App Service with Python runtime:
```
az webapp create --resource-group YourResourceGroup --plan YourAppServicePlan --name YourAppName --runtime "PYTHON|3.10"
```

2. Configure the application settings:
```
az webapp config appsettings set --resource-group YourResourceGroup --name YourAppName --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITE_PORT=8000 STARTUP_COMMAND="bash startup.sh"
```

3. Deploy the code using Git or GitHub Actions:
```
az webapp deployment source config --resource-group YourResourceGroup --name YourAppName --repo-url YourGitHubRepoURL --branch main --manual-integration
```

4. Important considerations for successful deployment:
   - The app uses heavy ML dependencies that require time to load
   - Increased timeout settings are in the startup.sh script
   - Consider using a higher tier App Service plan for adequate memory (at least P1v2)
   - First load may take a few minutes as the models are downloaded and initialized

## Using the Error Analysis Tool

The Error Analysis tool helps you understand where your sentiment model is failing on the SST-2 dataset. To use it:

1. Navigate to the "Error Analysis" page from the navigation bar
2. Set the number of samples to analyze (100-500) and the confidence threshold
3. Click "Analyze Model Errors" to start the analysis
4. Explore the results across different tabs:
   - **Performance Analysis**: View confusion matrix, confidence distribution, and sentence feature impacts
   - **Error Pattern Analysis**: Explore categorized error types with examples and descriptions
   - **Error Examples**: Browse specific high-confidence errors with word cloud visualization

5. For deeper analysis of specific errors, use the analysis tools:
   - **LIME Analysis**: Select an error example and click "Analyze with LIME" to see which words influenced the incorrect prediction
   - **Counterfactual Testing**: Select an error example and click "Test Counterfactuals" to see how small changes to the sentence affect the model's prediction
   - **Similarity Analysis**: Click "Analyze Error Similarity" to find clusters of similar errors and identify common patterns

## Understanding Error Patterns

The tool categorizes errors into several types:

- **Negation Errors**: When the model fails to properly handle negation words (not, n't, no, never)
- **Intensity Errors**: When the model misinterprets intensity modifiers (very, extremely, really)
- **Context Errors**: When the model struggles with longer sentences with complex context
- **Comparison Errors**: When the model fails on sentences with contrasting parts (but, however, although)
- **Sarcasm Errors**: When the model misses potential sarcasm or irony
- **Ambiguity Errors**: When the model struggles with words that have context-dependent sentiment
- **Conditional Errors**: When the model has difficulty with hypothetical or conditional statements

## Counterfactual Testing

Counterfactual testing helps identify the minimal changes needed to fix model errors:

1. For negation errors, it tries removing negation words
2. For intensity errors, it tries removing intensity modifiers
3. For comparison errors, it tries splitting the sentence at comparison words
4. For ambiguity errors, it tries replacing ambiguous words with clearer alternatives
5. For all error types, it tries simplifying the sentence

This approach, inspired by Errudite's rewriting capabilities, helps pinpoint exactly what causes the model to make mistakes.

## Similarity-Based Error Analysis

The similarity analysis feature helps identify patterns across error cases:

1. **TF-IDF Vectorization**: Converts error sentences into TF-IDF vectors
2. **Clustering**: Groups similar errors together using K-means clustering
3. **t-SNE Visualization**: Creates a 2D map of error cases showing their relationships
4. **Nearest Neighbors**: For each error, finds the most similar other errors
5. **Key Terms**: Identifies the most important terms for each error cluster

This helps identify broader patterns in model failures beyond individual error categories.

## References

- LIME: [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)
- SST-2 Dataset: [https://huggingface.co/datasets/nyu-mll/glue/viewer/sst2](https://huggingface.co/datasets/nyu-mll/glue/viewer/sst2)
- TinyBERT: [https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
- Azimuth: [https://github.com/ServiceNow/azimuth](https://github.com/ServiceNow/azimuth)
- Errudite: [https://github.com/uwdata/errudite](https://github.com/uwdata/errudite) 