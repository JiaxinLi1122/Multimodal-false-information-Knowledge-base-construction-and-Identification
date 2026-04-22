# Multimodal Misinformation Detection: Knowledge Base Construction and Identification

> A deep learning system that detects fake news on social media by jointly analyzing **text** and **images** using BERT and VGG-19, supporting both Twitter and Weibo platforms.

---

## Background

The rapid spread of misinformation on social media platforms poses serious challenges to public discourse and trust. Traditional text-only detection approaches are increasingly insufficient, as fabricated posts often pair misleading text with manipulated or out-of-context images.

This project addresses that gap with a **multimodal approach**: by fusing language understanding (BERT) and visual understanding (VGG-19), the model captures cross-modal inconsistencies that neither modality alone can reliably detect. The system covers both English (Twitter/FakeNewsNet) and Chinese (Weibo) content, along with dedicated crawlers for each platform.

---

## Features

- **Dual-platform support** — processes data from Twitter (English) and Weibo (Chinese)
- **Multimodal fusion** — jointly encodes text and images into a unified representation
- **Pretrained backbone models** — leverages BERT-base-uncased and VGG-19 with ImageNet weights
- **End-to-end pipeline** — from raw social media data to binary fake/real classification
- **Flexible data storage** — supports CSV, SQLite, MySQL, MongoDB, and Kafka export
- **Keyword analysis** — extracts and stores high-frequency terms from misinformation posts
- **Modular crawlers** — independent, configurable scrapers for Twitter API and Weibo

---

## Tech Stack

| Category         | Technology                                      |
|------------------|-------------------------------------------------|
| Deep Learning    | PyTorch                                         |
| NLP              | BERT (`bert-base-uncased`) via HuggingFace Transformers |
| Computer Vision  | VGG-19 via TorchVision                          |
| Data Processing  | Pandas, NumPy, PIL, scikit-image                |
| NLP Utilities    | NLTK (tokenization, lemmatization, stopwords)   |
| Twitter Crawler  | Twython, Newspaper3k                            |
| Weibo Crawler    | Requests + custom HTML parsers                  |
| API Rate Manager | Flask, Flask-CORS                               |
| Databases        | SQLite, MySQL, MongoDB                          |
| Messaging        | Apache Kafka (optional export)                  |

---

## Project Structure

```
.
├── main.py                  # Training entry point — full training loop
├── dataset.py               # PyTorch Dataset — loads CSV + images, BERT tokenization
├── mult_models.py           # Model definitions — TextEncoder, VisionEncoder, Fusion
├── test.py                  # Evaluation script — runs model on test set
├── key_words.py             # Keyword extraction — writes top-40 terms to SQLite
│
├── config/
│   ├── config.json          # Model hyperparameters (FC layer sizes, dropout)
│   └── config_opt.json      # Optimizer settings (learning rate, epsilon)
│
├── crawler/
│   ├── Twitter/
│   │   ├── main.py                      # Crawler entry (factory pattern, multiprocessing)
│   │   ├── config.json                  # Sources: politifact, gossipcop
│   │   ├── tweet_collection.py          # Bulk tweet fetching (100/request)
│   │   ├── news_content_collection.py   # News article scraping via Newspaper3k
│   │   ├── user_profile_collection.py   # User metadata collection
│   │   ├── resource_server/             # Flask app for multi-key API rate management
│   │   └── util/TwythonConnector.py     # Twitter API connector with auto rate-limit handling
│   │
│   └── weibo/
│       ├── spider.py          # Main Weibo crawler (date range, filters, downloads)
│       ├── weibo.py           # Weibo post data model
│       ├── user.py            # User data model
│       ├── parser/            # Page parsers (posts, comments, albums, photos)
│       ├── writer/            # Export writers (CSV, JSON, MySQL, SQLite, MongoDB, Kafka)
│       └── downloader/        # Image and video downloaders
│
└── data/
    ├── twitter/
    │   ├── train_posts_clean.csv    # ~13,366 labeled training posts
    │   ├── test_posts.csv           # ~1,111 test posts
    │   ├── images_train/            # 412 event-level image folders
    │   └── images_test/             # 106 event-level image folders
    │
    └── weibo/
        ├── text_content/            # Train/test splits for rumor & non-rumor
        ├── rumor_images/            # 82 rumor event image folders
        ├── nonrumor_images/         # 190 non-rumor event image folders
        ├── w2v.pickle               # Pretrained Word2Vec embeddings
        ├── word_embedding.pickle    # Word embedding matrix
        └── stop_words.txt           # Chinese stopword list
```

---

## Model Architecture

The model follows a **dual-stream fusion** design:

```
┌─────────────────────────────────────────────────────────┐
│  Text Stream                                            │
│                                                         │
│  Raw Text ──► BERT (bert-base-uncased)                  │
│               └── [CLS] token → 768-dim                 │
│               └── FC1 (768 → 2742) + ReLU + Dropout     │
│               └── FC2 (2742 → 32)                       │
│               └── 32-dim text feature vector            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Feature      │     64-dim concatenation
              │  Fusion Layer │──► FC (64 → 35) ──► FC (35 → 1) ──► Sigmoid
              └───────────────┘
                      ▲
┌─────────────────────┴───────────────────────────────────┐
│  Vision Stream                                          │
│                                                         │
│  Image (224×224) ──► VGG-19 (pretrained, ImageNet)      │
│                       └── 4096-dim feature              │
│                       └── FC1 (4096 → 2742) + ReLU + Dropout │
│                       └── FC2 (2742 → 32)               │
│                       └── 32-dim visual feature vector  │
└─────────────────────────────────────────────────────────┘

Output: sigmoid(score) > 0.5 → FAKE  |  ≤ 0.5 → REAL
```

**Design notes:**
- BERT and VGG-19 weights are frozen by default (`fine_tune = false`) to reduce memory requirements and prevent overfitting on limited data
- Dropout (p=0.4) is applied after each intermediate FC layer
- Loss function: Binary Cross-Entropy (`BCELoss`)
- Optimizer: AdamW with linear warmup scheduler (lr = 3e-5)

---

## Dataset

### Twitter (English)

| Split | File | Records |
|-------|------|---------|
| Train | `train_posts_clean.csv` | ~13,366 |
| Test  | `test_posts.csv`        | ~1,111  |

**Columns:** `post_id`, `post_text`, `user_id`, `image_id`, `username`, `timestamp`, `label` (fake/real)

**Images:** Organized by event (e.g., `boston_fake_*`, `attacks_paris_*`) — 412 train folders, 106 test folders.

**Sources:** [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) — PolitiFact and GossipCop.

### Weibo (Chinese)

| Split | Rumor | Non-Rumor |
|-------|-------|-----------|
| Train | 5,486 posts | 4,515 posts |
| Test  | 892 posts   | 918 posts   |

**Structure per post (3 lines):**
```
Line 1: tweet_id | username | tweet_url | user_url | timestamp | is_original |
        retweet_count | comment_count | like_count | user_id | verified |
        followers | following | tweet_count | platform
Line 2: image_url_1 | image_url_2 | null
Line 3: post_content
```

---

## Getting Started

### Prerequisites

- Python 3.7+
- CUDA-enabled GPU **strongly recommended** (CPU training will be extremely slow)
- ~8 GB VRAM minimum (16 GB recommended with both BERT and VGG-19 loaded)

### Install Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install pandas numpy
pip install Pillow scikit-image
pip install nltk
pip install tqdm
pip install flask flask-cors        # only needed for Twitter crawler
pip install twython newspaper3k     # only needed for Twitter crawler
pip install requests
```

Download required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

### Run

```bash
# 1. Train the model
python main.py

# 2. Evaluate on test set
python test.py

# 3. Extract misinformation keywords → writes to data/test.db
python key_words.py
```

To adjust model hyperparameters, edit `config/config.json`. To change the learning rate or optimizer settings, edit `config/config_opt.json`.

---

## Workflow

```
1. Data Collection
   ├── Twitter crawler  →  tweets + article text + images + user profiles
   └── Weibo crawler    →  posts + images + user metadata

2. Preprocessing
   ├── Text: BERT tokenization, max length 500 tokens
   └── Images: resize to 224×224, ImageNet normalization

3. Model Training  (main.py)
   ├── Batch size: 8
   ├── Epochs: up to 50
   ├── Loss: BCELoss
   └── Optimizer: AdamW + linear warmup

4. Evaluation  (test.py)
   └── Metrics: accuracy, per-class performance

5. Keyword Analysis  (key_words.py)
   └── Top-40 misinformation keywords → SQLite (data/test.db, table: news_kw)
```

---

## Results

The model outputs a probability score per post. With threshold 0.5:

- Score > 0.5 → classified as **FAKE**
- Score ≤ 0.5 → classified as **REAL**

Evaluation metrics include binary classification accuracy on the held-out test set. The multimodal fusion consistently outperforms text-only or image-only baselines by capturing cross-modal inconsistencies (e.g., text claiming an event while the image is from an unrelated context).

---

## Future Work

- [ ] Replace VGG-19 with Vision Transformer (ViT) or CLIP for stronger image-text alignment
- [ ] Add cross-attention between text and image features instead of simple concatenation
- [ ] Extend to multilingual BERT for unified Chinese/English modeling
- [ ] Build a REST API + web interface for real-time post verification
- [ ] Incorporate social graph features (retweet patterns, user credibility scores)
- [ ] Add explainability — highlight which words/image regions contributed to the prediction

---

## Contributing

Contributions, issues, and pull requests are welcome. Please open an issue first to discuss any significant changes.

---

## License

This project is intended for academic and research purposes.
