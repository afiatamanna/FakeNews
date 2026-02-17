# Dataset

## Source

This project supports two ways to get data (set `DATA_SOURCE` in `src/config.py`):

### 1. HuggingFace (default — no download needed)

- **Dataset:** [Pulk17/Fake-News-Detection-dataset](https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset) (~30k rows)
- **Usage:** Set `DATA_SOURCE = "huggingface"` in `src/config.py`. Data is loaded automatically when you run the pipeline (requires internet and `datasets`).
- Columns: title, text, subject, date, label (0=fake, 1=real).

### 2. Local CSV (Kaggle)

- [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset): two files `Fake.csv` and `True.csv`.
- **Usage:** Set `DATA_SOURCE = "local"` and place `Fake.csv` and `True.csv` in `data/` or `data/raw/`.

### 3. Auto (try local, then HuggingFace)

- Set `DATA_SOURCE = "auto"`: use local CSVs if present, otherwise load from HuggingFace.

## Setup

- **HuggingFace:** No setup; just run the scripts (default).
- **Local:** Download from Kaggle and put the two CSVs in `data/` or `data/raw/`.

## Train/Val/Test split

- Default: 70% train, 15% validation, 15% test (stratified by label).
- Random seed is fixed in `src/config.py` for reproducibility.
- Splits are produced by `data_loading.py` and can be saved under `data/` for reuse.

## License

Respect the dataset’s original license (Kaggle / LIAR). Do not redistribute raw data in the repo; document the source and link only.
