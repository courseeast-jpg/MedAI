# MedAI v1.1 - Gemini API Setup Guide

## Step 1: Get Gemini API Key (5 minutes)

1. Go to: https://ai.google.dev/
2. Click "Get API key in Google AI Studio"
3. Sign in with your Google account
4. Click "Create API key"
5. Copy the key (starts with `AIza...`)

**Free Tier:**
- 1,500 requests/day
- No credit card required
- Never expires

## Step 2: Configure MedAI

Open `.env` file in medai folder and update:

```
GEMINI_API_KEY=AIzaSy...your_actual_key_here...
DB_ENCRYPTION_KEY=choose_a_strong_passphrase_here
```

Save the file.

## Step 3: Install Dependencies

```cmd
cd "G:\MEDICAL\Med AI\medai_v1.1_validated\medai"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Step 4: Initialize Databases

```cmd
python scripts\init_db.py
python scripts\init_chroma.py
```

## Step 5: Launch MedAI

```cmd
streamlit run app\main.py
```

Opens: http://localhost:8501

## What's Changed

- **AI Engine:** Gemini 2.5 Flash instead of Claude
- **Cost:** Free (1,500 requests/day)
- **Quality:** ~85% of Claude Sonnet for medical text extraction
- **Fallback:** spaCy rules-based extraction if API unavailable

## Testing

Upload a medical PDF to test:
- PDF text extraction (docling)
- PII stripping (Presidio)
- Entity extraction (Gemini API)
- Knowledge base storage (SQLite + ChromaDB)

## Troubleshooting

**"No module named google.generativeai"**
→ Run: `pip install google-generativeai`

**"Invalid API key"**
→ Check `.env` file, ensure GEMINI_API_KEY is correct

**"Rate limit exceeded"**
→ Free tier: 1,500 req/day. Wait or upgrade to paid tier.

## Next Steps

After successful test:
1. Run golden tests: `pytest tests\golden\ -v`
2. Upload real medical PDFs
3. Review extracted entities in UI
4. Monitor API usage in Google AI Studio dashboard
