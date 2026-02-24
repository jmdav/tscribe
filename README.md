# tScribe

Simple Python script with streamlit GUI for batch transcribing audio files.

## Installation

```bash
pip install -r requirements.txt
```

### Building the Clickable Diff Component

The Review tab uses a custom Streamlit component for clickable diff text. The component is pre-built, but if you need to rebuild it:

```bash
cd components/clickable_diff/frontend
npm install
npm run build
```

## Usage

```bash
streamlit run app.py
```
