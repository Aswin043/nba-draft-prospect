
# NBA Draft 2025 – Player Archetypes & Similarity

Clusters players from college stats and lets you query similar players.

## Build (optional)

```bash
python archetypes.py --input nba_draft_2025_final.csv --output archetypes_prepared.csv --algo kmeans --k 4
```

## App

```bash
pip install -r requirements.txt
streamlit run app.py
```
