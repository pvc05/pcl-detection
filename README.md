# pcl-detection
nlp coursework to detect patronising and condescending language (pcl) at imperial


### Directory Structure
The following diagram illustrates the organisation of the repository:

```
pcl-detection/
|-- BestModel.ipynb                 # Final clean training pipeline & evaluation
|-- data_analysis.ipynb             # Exploratory Data Analysis (EDA)
|-- models.ipynb                    # Initial model experimentation & grid search
|-- span_model.ipynb                # Optuna hyperparameter optimization
|-- dev.txt, test.txt               # Final submission prediction files
|-- data/                           # Dataset files (train, dev, test, categories)
|   |-- dontpatronizeme_*.tsv       # Raw data source files
|   |-- *_semeval_*.csv             # Standardized split definitions
|-- models_cache/                   # Cached HuggingFace models (not commited on repo)
|-- pcl_tf/                         # Custom Python module for model components
|   |-- span_tf.py                  # SpanModel architecture & MultiTaskTrainer
|   |-- focal_loss.py               # Focal Loss implementation
|   |-- dataset_manager.py          # Dataset classes (SpanDS)
|   |-- tf.py                       # Old model architecture (archive)
|-- search_results/                 # All search results from grid search
|   |-- best_hyperparams.json       # Optimal hyperparameters for best f1
|   |-- ...
```

### File Highlights

*   **BestModel.ipynb**: The primary notebook for the final model. It implements the complete training pipeline using the best hyperparameters found. It handles data loading, weighted sampling, multi-task training, threshold optimization, and generates the final submission files (`dev.txt` and `test.txt`).
*   **span_model.ipynb**: Dedicated to hyperparameter optimization using Optuna. It searches for optimal values for learning rate, dropout, focal loss parameters ($\alpha, \gamma$), and auxiliary loss weights without data leakage.
*   **models.ipynb**: Used for initial model exploration and grid search experiments. It tests various architectural modifications and transformer backbones (like RoBERTa, DeBERTa) before settling on the final approach.
*   **data_analysis.ipynb**: Contains detailed Exploratory Data Analysis (EDA) of the dataset, including class distribution analysis, keyword analysis, and Principal Component Analysis (PCA) of text features.
*   **pcl_tf/**: A custom Python module containing the core logic:
    *   `span_tf.py`: Defines the `SpanModel` architecture (Encoder + Binary Head + Category Head) and the `MultiTaskTrainer`.
    *   `focal_loss.py`: Implements the Focal Loss function to address the severe class imbalance in the dataset.
    *   `dataset_manager.py`: Manages data loading and tokenization via the `SpanDS` class.
*   **search_results/best_hyperparams.json**: A JSON file storing the optimal hyperparameters discovered by Optuna, which are loaded by `BestModel.ipynb` for reproducibility.