# 20 Newsgroups Text Classification with DistilBERT

This repository contains a PyTorch implementation for **text classification** on the widely used **20 Newsgroups dataset**, leveraging the **DistilBERT** pre-trained language model from the Hugging Face `transformers` library. The model is fine-tuned to categorize newsgroup posts into one of the 20 available topics.

## 🚀 Project Overview

The core task is to classify raw text documents (newsgroup posts) into their corresponding categories. This project demonstrates the workflow for:
1.  Loading the 20 Newsgroups dataset.
2.  Tokenizing the text using the DistilBERT tokenizer.
3.  Creating a custom PyTorch `Dataset` class.
4.  Fine-tuning a `DistilBertForSequenceClassification` model.
5.  Evaluating the model using accuracy.
6.  Making predictions on new, unseen text.

## 📋 Requirements

To run the code, you'll need Python and the following libraries:

* `torch` (PyTorch)
* `transformers` (Hugging Face)
* `scikit-learn`
* `numpy`

## 📂 Dataset

The project uses the **20 Newsgroups dataset**, which is automatically loaded using `sklearn.datasets.fetch_20newsgroups`.

  * **Total Samples:** 18,846 newsgroup documents.
  * **Topics (Labels):** 20 distinct categories (e.g., `comp.graphics`, `rec.sport.baseball`, `talk.politics.mideast`, etc.).
  * **Preprocessing:** The headers, footers, and quotes are removed from the posts to focus on the main content, as defined by the loading parameters: `remove=('headers', 'footers', 'quotes')`.
  * **Split:** The data is split into **80% training** and **20% testing** sets (`test_size=0.2`).

## ⚙️ Model and Tokenizer

| Component | Class/Model Name | Source | Purpose |
| :--- | :--- | :--- | :--- |
| **Tokenizer** | `AutoTokenizer` | `distilbert-base-uncased` | Converts text into token IDs, attention masks, and token type IDs. |
| **Model** | `AutoModelForSequenceClassification` | `distilbert-base-uncased` | DistilBERT fine-tuned for a 20-class sequence classification task. |

## 💡 Training Details

The model is fine-tuned using the **Hugging Face `Trainer`** class.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | `distilbert-base-uncased` | Base pre-trained model checkpoint. |
| **Number of Labels** | 20 | Matches the number of newsgroup categories. |
| **Learning Rate** | $2 \times 10^{-5}$ | Common setting for fine-tuning BERT-style models. |
| **Epochs** | 3 | Number of complete passes over the training data. |
| **Batch Size** (Train/Eval) | 8 / 8 | Batch size for training and evaluation. |
| **Weight Decay** | 0.01 | L2 regularization applied to all layers except bias and layer norm weights. |
| **Evaluation Metric** | **Accuracy** | Computed using `sklearn.metrics.accuracy_score`. |
| **Output Directory** | `./results` | Location where model checkpoints are saved (`save_strategy="epoch"`). |

## 🧩 Key Code Components

### `NewsGroupDataset` Class

A custom PyTorch `Dataset` handles the preparation of data for the model:

```python
class NewsGroupDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # ...
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Returns a dictionary containing token inputs and the label for a given index
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx])}
```

### `compute_metrics` Function

Used by the `Trainer` to evaluate the model's performance on the test set:

```python
def compute_metrics(p):
    # p is a named tuple (predictions, label_ids, metrics)
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}
```

## 🎯 Example Prediction

The notebook includes a final step demonstrating how to use the trained model (specifically, the one loaded before training in cell 4, which is technically untrained/randomly-initialized) to make a prediction on a sample text:

| Sample Text | Predicted Topic |
| :--- | :--- |
| `"The government passed a new law affecting international trade."` | `talk.religion.misc` |

*Note: The prediction shown in the notebook is from the **un-trained** or **randomly initialized** DistilBERT model. For accurate classification, the `trainer.train()` method must be executed first to fine-tune the model on the `train_dataset`.*

## 💻 How to Run

1.  Clone this repository.
2.  Ensure you have the required libraries installed (see **Requirements**).
3.  Execute the provided Jupyter Notebook: `text_class.ipynb`.
4.  To train the model, you'll need to run the `trainer.train()` command, which is currently missing from the notebook but implied by the setup.

-----
