# Fake News Detection Using Multiple Classifiers and Explainable AI

**Short research report — PhD portfolio project**

---

## Abstract

We address the problem of automatically classifying news articles as real or fake using a variety of machine learning approaches. We compare classical models (logistic regression, SVM, random forest) with deep learning (LSTM) and a pre-trained transformer (DistilBERT), all trained on a public fake/real news dataset. We further apply explainability methods (SHAP and LIME) to the best-interpretable pipeline (TF-IDF + logistic regression) to highlight which features drive predictions. Results are reported in terms of accuracy, precision, recall, F1, and confusion matrices. We discuss limitations, ethical considerations, and directions for future work.

**Keywords:** fake news detection, NLP, text classification, explainable AI, SHAP, LIME, BERT.

---

## 1. Introduction

The spread of false information online poses risks to public discourse and trust. Automated fake news detection is an active research area at the intersection of natural language processing (NLP) and machine learning (ML). This project has two main goals: (1) to compare several classification approaches—from simple linear models to transformer-based ones—on a binary real/fake task, and (2) to add interpretability so that users can understand *why* a given article was classified as real or fake.

We use a single, widely used dataset (Fake and Real News from Kaggle) and a fixed train/validation/test split. We evaluate all models with the same metrics and then apply SHAP and LIME to one classical pipeline to illustrate explainable AI (XAI) in this context.

---

## 2. Related Work

Fake news detection has been studied with hand-crafted features (e.g. n-grams, readability, source metadata) and with neural and pre-trained language models. Recent work often uses BERT or similar transformers and reports strong performance. Explainability in NLP has been addressed with attention visualization, gradient-based methods, and model-agnostic tools such as LIME and SHAP, which we adopt here for a TF-IDF + logistic regression model to keep explanations tractable and interpretable.

---

## 3. Methodology

### 3.1 Data

- **Source:** Fake and Real News dataset (Kaggle): two CSV files (Fake.csv, True.csv) with title, text, subject, and date.
- **Labels:** Binary — 0 (fake), 1 (real).
- **Preprocessing:** Lowercasing, URL removal, non-alphanumeric stripping, whitespace normalization. A cleaned text field is used for TF-IDF and LSTM; the same (or raw) text is used for BERT.
- **Split:** 70% train, 15% validation, 15% test (stratified), with a fixed random seed for reproducibility.

### 3.2 Models

1. **Classical ML:** TF-IDF (unigrams + bigrams, sublinear scaling) with:
   - Logistic regression
   - Linear SVM
   - Random forest
2. **LSTM:** Word-level vocabulary from training set, embedding + LSTM + linear layer (PyTorch).
3. **BERT:** Fine-tuned DistilBERT for sequence classification (HuggingFace). Same train/val/test split.

### 3.3 Evaluation

- **Metrics:** Accuracy, macro-averaged precision, recall, and F1.
- **Confusion matrix** for each model on the test set.
- Results are collected in a single comparison table (e.g. `outputs/results.csv`).

### 3.4 Explainability

- **SHAP:** LinearExplainer on the logistic regression model with TF-IDF features; summary plot of mean absolute SHAP values per feature.
- **LIME:** LimeTextExplainer on the same pipeline; local explanations for a few test examples, showing which words support “Fake” vs “Real”.

---

## 4. Results

(After running the pipeline, insert here the comparison table from `outputs/results.csv` and, if desired, one or two confusion matrix heatmaps.)

**Example table (fill with actual numbers):**

| Model               | Accuracy | Precision | Recall | F1   |
|---------------------|----------|-----------|--------|------|
| logistic_regression | …        | …         | …      | …    |
| svm                 | …        | …         | …      | …    |
| random_forest        | …        | …         | …      | …    |
| lstm                | …        | …         | …      | …    |
| bert                | …        | …         | …      | …    |

**Explainability:** The SHAP summary plot (`outputs/figures/shap_summary.png`) shows which TF-IDF features most influence the logistic regression output. The LIME figures (`outputs/figures/lime_example_*.png`) show example articles with highlighted words that push the prediction toward Fake or Real. Including 2–3 such figures in the report helps illustrate why explainability matters for trust and debugging.

---

## 5. Discussion

- **Model comparison:** Typically, BERT (or DistilBERT) achieves higher accuracy/F1 than classical models and LSTM on this dataset, at the cost of compute and less direct interpretability. Classical models are fast and amenable to SHAP/LIME.
- **Limitations:** (1) Single dataset and language (English); (2) binary setup—real-world veracity is often graded; (3) possible bias by topic or source; (4) temporal and domain shift if applied to new time periods or outlets.
- **Ethics:** Automated labels should not be the sole basis for content moderation; explainability can support human review and fairness checks.

---

## 6. Conclusion

We implemented a full pipeline for fake news detection using classical ML, LSTM, and BERT, and added explainability via SHAP and LIME on a logistic regression classifier. The project demonstrates comparison of methods, rigorous evaluation, and awareness of limitations and ethics—all relevant for PhD-level research.

**Research significance.** This work is intended as a foundation for further research rather than a finished product. Natural PhD-level extensions include: (1) **Explainable AI for transformers** — bringing SHAP/LIME or newer methods to BERT so that high-accuracy models become interpretable for moderators; (2) **Bias and fairness** — analysing and mitigating performance gaps across topics, sources, or demographic proxies; (3) **Domain and temporal shift** — studying how models degrade when applied to new time periods or outlets, and developing adaptation strategies; (4) **Multilingual and low-resource** settings; (5) **Fine-grained veracity** (e.g. LIAR) and uncertainty quantification. Any of these directions could form the core of a studentship or thesis.

---

## References

(Add 3–5 key papers on fake news detection and/or explainable NLP as you refine the report.)
