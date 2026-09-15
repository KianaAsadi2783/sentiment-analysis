# Sentiment Analysis

A Python-based sentiment analysis project using the **IMDb movie reviews dataset**.

## Overview

This project implements a sentiment analysis system that processes movie reviews and predicts their sentiment.

The project was developed as part of a university coursework project and uses the IMDb dataset (`aclImdb`) for training and evaluation.

## Dataset

The project uses the **IMDb Large Movie Review Dataset**, which contains labeled movie reviews for binary sentiment classification.

The dataset is **not included in this repository** because of its size.

To run the project, download the IMDb dataset separately and place the extracted `aclImdb` directory in the project root:

```text
sentiment-analysis/
├── aclImdb/
├── data.py
├── model.py
└── main.py
```

## Project Structure

```text
sentiment-analysis/
├── data.py             # Data loading and preprocessing
├── model.py            # Sentiment analysis model
├── main.py             # Main program
├── Documentation.pdf   # Project documentation
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Requirements

* Python 3.x
* Required Python packages listed in `requirements.txt`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

After installing the required dependencies and placing the IMDb dataset in the project directory, run:

```bash
python main.py
```

## Dataset Structure

The expected dataset structure is:

```text
aclImdb/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```

## Documentation

Additional information about the project can be found in:

**Documentation.pdf**

## Notes

The IMDb dataset is not included in this repository due to its size.

You can download it from the official Stanford AI Lab website:

[IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

Or download the dataset directly:

[Download `aclImdb_v1.tar.gz`](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

---

## Authors

**Kiana Asadi**

GitHub: [KianaAsadi2783](https://github.com/KianaAsadi2783)

**Maryam Motallebi**

GitHub: [mrym-mtlb](https://github.com/mrym-mtlb)

**Arshia Momtazi**

GitHub: [Arshiamtz](https://github.com/Arshiamtz)
