*Senticall – Voice & Text Sentiment Analytics (Main App + Report Generator)

End-to-end customer call analytics pipeline:

* **Part 1 (`main.ipynb`)**: Gradio app for analyzing **audio or text** calls, extracting structured insights (sentiment, urgency, keywords, solutions, safety flags), and saving results to a JSONL log.
* **Part 2 (`code_report.py`)**: Data processing + analytics dashboard + AI executive summary + strategic recommendations + automated **PowerPoint report** generation.

---

## Features

### Part 1 — Real-time Call Analysis (`main.ipynb`)

* Audio upload / microphone recording **or** text input
* **Prefers text** automatically when text is provided (even if audio exists)
* Audio pipeline:

  * Speech-to-text + translation to English
  * **Voice sentiment + emotions** inferred from audio prosody (pitch/pace/pauses)
* Text pipeline:

  * Structured sentiment + keywords + urgency + solutions
* Safety & guardrails:

  * Prompt-injection / privacy attempt detection
  * Safety keyword validation to avoid false positives
* Persistence:

  * Appends each analyzed call into **JSONL** storage (`agent_results.json`)
  * Displays a **history table** in the UI
* Controls:

  * New Call (new conversation ID, preserve history)
  * Clear History (reset state)
  * Generate Report (runs report script output)

---

### Part 2 — Report Pipeline (`code_report.py`)

* Converts JSONL → flattened CSV
* Merges with `customer_demographics.csv` on `phone`
* Feature engineering:

  * `ageGroup` bins
  * `incomeGroup2` recoding
* Analytics dashboard (Gradio):

  * Overall sentiment doughnut chart
  * Demographic bar charts (gender/age/income)
  * Time-trend line charts (month/day/hour)
  * Optional significance letters (z-test)
* AI layers (LangChain + OpenAI):

  * Executive summary (4–6 sentences)
  * Strategic recommendations (staffing, demographics, friction areas, priorities)
* Automated PPTX output:

  * Charts embedded as images
  * Summary + recommendations split into bullet slides (max 4 bullets/slide)
  * Numbered titles (e.g., “Executive Summary (1)”)

---

## Project Structure

```txt
.
├── main.ipynb                 # Part 1: Gradio app (voice + text analysis)
├── code_report.py             # Part 2: report pipeline + dashboard + PPTX generator
├── customer_demographics.csv  # Input demographics (required for merging)
├── agent_results.json         # Generated JSONL log (created by Part 1)
├── requirements.txt
└── README.md
```

---

## Requirements

* Python 3.10+ recommended
* Google Colab recommended (because:

  * `google.colab.userdata`
  * `google.colab.files.download`
  * microphone/audio UI works smoothly)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## API Keys / Secrets

### Required

* `GOOGLE_API_KEY` (Gemini)
* `LANGCHAIN_API_KEY` (LangSmith tracing; optional but supported)
* `OPENAI_API_KEY` (only needed in Part 2 for executive summary + recommendations)

### In Google Colab

Add keys in **Colab → Secrets**:

* `GOOGLE_API_KEY`
* `LANGCHAIN_API_KEY` (optional)
* `OPENAI_API_KEY` (for Part 2 AI summary/recommendations)

---

## How to Run

### 1) Run the Main App (Part 1)

Open `main.ipynb` in Colab and run all cells.

The Gradio UI provides:

* Phone validation + button enable/disable
* Audio or text input
* Analysis + persistent history
* Export JSONL
* Generate Report button (calls the report pipeline)

Output file created:

* `agent_results.json` (JSONL format)

---

### 2) Run the Report Pipeline (Part 2)

Ensure these files exist in your runtime working directory:

* `code_report.py`
* `agent_results.json` (or the JSONL file name referenced in your script)
* `customer_demographics.csv`

Then run:

```bash
python code_report.py
```

Outputs (typical):

* `agent_results_flattened.csv`
* `merged.csv`
* `merged_with_ageGroup.csv`
* chart images (`chart_*.png`)
* PowerPoint report (e.g., `Agent_10x_Report_Numbered.pptx`)

---

## Input Data Format

### `agent_results.json` (JSONL)

One JSON object per line. Minimal structure:

```json
{
  "id": "CONV-YYYYMMDD-HHMMSS",
  "phone": "123456789",
  "date": "YYYY-MM-DD",
  "time": "HH:MM:SS",
  "data": {
    "sentiment": "Positive|Neutral|Negative|Mixed",
    "summary": "...",
    "solutions": ["...", "..."],
    "priority_score": 1
  }
}
```

### `customer_demographics.csv`

Must include a `phone` column. Example columns:

* `phone`, `age`, `gender`, `incomeGroup`, `education`, ...

---

## Notes / Common Issues

* **Datetime parsing**: Part 2 uses:

  ```python
  format="%d/%m/%Y %H:%M:%S"
  ```

  Make sure your `date` + `time` match this format. If your date is saved as `YYYY-MM-DD`, update the format accordingly.
* **Colab-specific imports** (`google.colab.*`) will fail locally unless removed or guarded.
* **PPTX generation** requires chart functions to return Matplotlib figures (already supported).

---

## Roadmap Ideas

* Add interactive filters (date range, demographic selector) in the dashboard
* Add export buttons for merged datasets and chart images
* Add per-call drill-down (select a Call ID and render full details)
* Add caching for expensive computations (plots, AI summaries)

---

## License

Add your preferred license here (MIT / Apache-2.0 / proprietary).

