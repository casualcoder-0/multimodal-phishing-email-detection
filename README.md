# PhishLens: Multimodal, Real-Time and Explainable AI Framework for Phishing Email Detection

## Quick setup on Windows (Python 3.11 recommended)

1. Extract the ZIP. Open the extracted folder in VS Code.
2. Open Terminal > New Terminal.
3. Run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.11` fails, install Python 3.11 or use `py -m venv .venv`.

## Download a real labelled dataset

Download `Phishing_Email.csv` from the Hugging Face repository:
https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/blob/main/Phishing_Email.csv

Click Download and place it at `data/Phishing_Email.csv`.

The dataset is a third-party collection. Inspect its license and labels before academic use. It provides email text and a label; it does **not** establish full multimodal validation by itself.

## Train and launch

```powershell
.venv\Scripts\python.exe train_model.py --data data/Phishing_Email.csv --max-rows 20000 --mode baseline
.venv\Scripts\python.exe -m streamlit run app.py
```

Visit the local URL printed by Streamlit (usually http://localhost:8501).

**For a genuinely trained hybrid model**, use a dataset containing labelled email text and meaningful URL, sender, or HTML fields. If those are available:

```powershell
.venv\Scripts\python.exe train_model.py --data data/your_richer_dataset.csv --mode hybrid
```

Important: If your dataset only has text and labels, training `--mode hybrid` will extract URL features from the text but missing sender/HTML metadata will remain empty. This is **not** evidence of independently validated sender or HTML detection. Compare with baseline before claiming improvement.

## What the project actually does

- Real ML: TF-IDF + Logistic Regression trained on labelled emails.
- Optional early feature fusion: TF-IDF plus numeric URL, sender and HTML features when source data supports them.
- On-demand inference: no model retraining per request; displays measured prediction time.
- Explainability: top TF-IDF feature contributions plus separate heuristic warning indicators.
- Safe URL parsing: URLs are inspected but never fetched.
- Evaluation: held-out accuracy, precision, recall, F1, and confusion matrix saved in `results/metrics.json`.

## Academic caveats

- The basic dataset is text-only. It supports a **baseline**, not a validated full multimodal claim.
- Test metrics are produced by your actual local training run. Never invent results.
- A random train/test split can overestimate real-world generalization. A separate newer dataset is needed for stronger validation.
- The score is not calibrated probability. False negatives and false positives are possible.
- This is a research prototype, not a production email security gateway.
- Avoid training on sensitive private inbox data without permission.

## Phase 3 form, after running successfully

**Implementation:** Trained a TF-IDF + Logistic Regression classifier and connected it to a Streamlit interface for email classification. Added URL and metadata inspection as separate warning indicators. [Only claim a trained multimodal model if you ran and evaluated hybrid mode on suitable data.]

**Architecture:** Email input -> preprocessing and feature extraction -> trained classifier -> on-demand inference -> feature contributions and warnings. Validated using a held-out test split.

**Testing:** Copy the actual values from `results/metrics.json`. State the number of test emails and any observed false positives/negatives.

## Troubleshooting

- `ModuleNotFoundError`: use `.venv\Scripts\python.exe` consistently.
- `Dataset not found`: verify CSV filename and path.
- `Unknown labels`: inspect your CSV's label column and edit mapping in `normalize()`.
- `No trained model`: run training before Streamlit.
- Very slow training: try `--max-rows 5000` first.
