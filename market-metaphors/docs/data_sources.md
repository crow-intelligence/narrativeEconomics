# Data Sources for Narrative Economics Analysis

Datasets for detecting metaphorical language in economic/financial discourse, mapped to
Kindleberger's crisis phases (displacement, boom, euphoria, distress, revulsion) and
Shiller's narrative economics framework.

## Status Overview

| Dataset | Coverage | Size | Text depth | Status |
|---------|----------|------|------------|--------|
| FOMC Meeting Minutes | 1992--2026 | 12 MB | Full minutes | Downloaded |
| FOMC Paragraph-Level Docs | Historical | 327 MB | 9 doc types, paragraph-level | Downloaded |
| Kaggle Stock News (full text) | 2009--2020 | ~1.76M URLs | Lead + body (scraped) | In progress |
| HuffPost News Sentiment | 2012--2022 | 95 MB | Headline + lead | Downloaded |
| CBS Central Bank Speeches | 1986--2023 | 498 MB, 35K speeches | Full speech text | Downloaded |
| BIS Speeches Archive | 1996--2025 | 366 MB | Full speech text | Downloaded |
| ECB-FED Speeches | 1996--2025 | 140 MB, 4,987 rows | Full text + OCR | Downloaded |
| RBA Board Minutes | 2015--2026 | 1.2 MB, 118 docs | Full minutes | Downloaded |
| BoC Deliberations | 2025--2026 | 0.1 MB, 10 docs | Full text | Downloaded |
| BoE MPC Minutes | 2015+ | TBD | Full minutes | Deferred (site blocks scraping) |
| Financial PhraseBank | Financial news | ~2 MB, 4,840 sentences | Annotated opinion sentences | To do |
| FiQA | Financial forums | ~50 MB | Opinion mining + QA | To do |
| Twitter Financial Sentiment | Financial tweets | ~10 MB | Tweets + sentiment | To do |
| Reddit Finance | 2008--2025 | ~500 MB--1 GB | Posts with selftext | To do |
| Strux Earnings Calls | 2017--2024 | ~1--2 GB, 11,950 transcripts | Full transcripts | To do |
| Multisource (selected subsets) | 1990--2025 | ~3--5 GB filtered | Full articles (bloomberg, reddit, all_the_news) | To do |
| EDGAR-CORPUS (MD&A) | 1993--2020 | Large | Management commentary | To do (deferred) |
| MNB (Hungarian) | 2011+ | TBD | PDFs (Hungarian) | Future work |
| FNSPID | 1999--2023 | 29.6 GB | Null (broken dataset) | Skipped (unusable) |

---

## 1. US Federal Reserve

The Fed's communications are the densest source of economic metaphors in English.
FOMC Minutes contain carefully worded assessments where language choices carry
policy signals. The Beige Book uses anecdotal, regional language rich in everyday
metaphors (e.g. "the economy is overheating", "a soft landing").

### 1.1 FOMC Meeting Minutes

**Status: Downloaded** -- `data/raw/fed/fomc-minutes.csv`

| | |
|---|---|
| **Source** | https://www.kaggle.com/datasets/ganghyeoklee/fomc-meeting-minutes-auto-updated |
| **Coverage** | 1992--2026 (267 meetings, auto-updates after each FOMC meeting) |
| **Size** | 12 MB |
| **Format** | CSV: `Date`, `Text` (full minutes), `Chair` (Greenspan / Bernanke / Yellen / Powell) |
| **License** | MIT |

The `Chair` column enables tracking how metaphor framing shifts across Fed leadership eras.

```bash
kaggle datasets download -d ganghyeoklee/fomc-meeting-minutes-auto-updated --unzip -p data/raw/fed/
```

### 1.2 FOMC Paragraph-Level Documents

**Status: Downloaded** -- `data/raw/fed/fomc_documents.csv` + `data/raw/fed/documents_by_type/`

| | |
|---|---|
| **Source** | https://www.kaggle.com/datasets/edwardbickerton/fomc-text-data |
| **Scraper** | https://github.com/edwardbickerton/Fed-Scraper |
| **Coverage** | Historical Fed materials (5+ years old) |
| **Size** | 327 MB main CSV + 9 per-type CSVs |
| **Format** | CSV: `document_kind`, `meeting_date`, `release_date`, `url`, text content |
| **Doc types** | Agendas, bluebooks, greenbooks, meeting minutes, meeting transcripts, policy statements, press conference transcripts, redbooks, miscellaneous |
| **License** | Not specified |

Paragraph-level granularity enables metaphor density analysis per document section.
The Beige Book portions are particularly valuable -- anecdotal regional summaries
use far more everyday metaphors than formal policy language.

```bash
kaggle datasets download -d edwardbickerton/fomc-text-data --unzip -p data/raw/fed/
```

### 1.3 FedTools (deprecated)

Python package (https://github.com/David-Woroniuk/FedTools) -- last updated Sep 2020.
Not recommended; the Kaggle datasets above supersede it.

---

## 2. International Central Banks (English)

Cross-country comparison reveals whether metaphor patterns during crises are universal
(supporting Shiller's narrative contagion thesis) or culturally specific. Speeches
are particularly valuable because they target public audiences and use more accessible,
metaphor-laden language than formal policy documents.

### 2.1 CBS Dataset -- 131 Central Banks

**Status: Downloaded** -- `data/raw/speeches/cbs_speeches.csv`

| | |
|---|---|
| **Source** | https://cbspeeches.com/ |
| **Codebook** | https://cbspeeches.com/CBSpeeches_Codebook.pdf |
| **Coverage** | 1986--2023, 35,487 speeches from 131 central banks |
| **Format** | CSV, JSON |
| **Includes** | ECB, BoE, BoC, RBA, Fed, and 126 others |
| **License** | Free for academic use |

The single highest-value download remaining. One file gives us speech text from every
major English-speaking central bank, enabling direct cross-country metaphor comparison
during shared crises (2008 GFC, 2011 EU debt, 2020 COVID).

### 2.2 BIS Speeches Archive

**Status: Downloaded** -- `data/raw/speeches/bis/speeches.csv`

| | |
|---|---|
| **Source** | https://www.bis.org/cbspeeches/download.htm |
| **Coverage** | 1996--present (more recent than CBS) |
| **Format** | Pre-compiled full-text extracts |
| **Scraper** | https://github.com/entelecheia/bis-fetcher (Python) |

Complements the CBS dataset with more recent speeches (2023--present).

### 2.3 ECB Speeches & Press Conferences

**Status: Downloaded** -- `data/raw/speeches/ecb_fed/`

| | |
|---|---|
| **Speeches (HuggingFace)** | https://huggingface.co/datasets/istat-ai/ECB-FED-speeches (1996--2025, OCR-extracted) |
| **Press conferences** | Academic dataset on ScienceDirect (1998--2016, 205 docs with statements + Q&A) |
| **Official download** | https://www.ecb.europa.eu/press/key/html/downloads.en.html |

```python
from datasets import load_dataset
ds = load_dataset("istat-ai/ECB-FED-speeches")
```

### 2.4 Bank of England -- MPC Minutes

**Status: Deferred** (site blocks automated access, dynamic page loading)

| | |
|---|---|
| **MPC Minutes** | https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/ |
| **Transcripts** | From 2015+ (8-year publication delay) |
| **Schedule** | 8 times per year |
| **Format** | HTML / PDF |
| **Scraping guide** | https://aurelien-goutsmedt.com/post/scraping-boe/ |

BoE speeches are already covered by CBS/BIS above; this adds the formal MPC meeting records.

### 2.5 Bank of Canada -- Speeches & MPR

**Status: Downloaded** (deliberations) -- `data/raw/boc/deliberations.parquet`. Speeches covered by CBS/BIS.

| | |
|---|---|
| **Speeches** | https://www.bankofcanada.ca/press/speeches/ |
| **MPR text analytics** | https://github.com/bankofcanada/MPR-Text-Analytics-2019 (code + data) |
| **Format** | HTML |

BoC speeches are already covered by CBS/BIS above; this adds Monetary Policy Report text.

### 2.6 Reserve Bank of Australia -- Board Minutes

**Status: Downloaded** -- `data/raw/rba/minutes.parquet` (118 docs, 2015--2026)

| | |
|---|---|
| **Board minutes** | https://www.rba.gov.au/monetary-policy/rba-board-minutes/ (Oct 2006+) |
| **Speeches** | https://www.rba.gov.au/speeches/ (text + audio + Q&A) |
| **Format** | HTML |

RBA speeches are already covered by CBS/BIS above; this adds the formal board minutes.

---

## 3. Financial News

News articles provide the public-facing narrative layer -- how journalists and analysts
frame market events for retail audiences. The editorial lead (first paragraph) is where
metaphor density tends to be highest, as writers "sell" the story.

### 3.1 Kaggle Stock News -- Full Article Text

**Status: In progress** -- `data/raw/articles/{domain}.parquet`

| | |
|---|---|
| **Source** | https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests |
| **Coverage** | 2009--2020, ~3.25M rows, ~1.76M unique URLs |
| **Domains** | benzinga.com (54%), seekingalpha.com (22%), zacks.com (12%), gurufocus.com, investors.com, and 11 others |
| **Format** | Per-domain parquet: `url`, `domain`, `lead`, `body`, `status`, `content_hash`, `fetch_date` |
| **Tool** | trafilatura (article extraction) |
| **Runtime** | ~30--60 hours at 1.5s delay per domain |

Scraper running via `uv run python scripts/pipeline/00_fetch_articles.py --delay 1.0`.
Crash-recoverable -- checkpoints every 100 articles per domain, skips already-fetched
URLs on restart. Articles are split into lead (first paragraph) and body (rest).

### 3.2 HuffPost News Sentiment

**Status: Downloaded** -- `data/raw/huffpost/data_20{12..22}.json`

| | |
|---|---|
| **Source** | https://www.kaggle.com/datasets/irakozekelly/financial-news-sentiment-dataset-20122022 |
| **Mirror** | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVW7SF |
| **Coverage** | 2012--2022, ~200K rows |
| **Size** | 95 MB (11 JSON files, one per year) |
| **Format** | JSON: `headline`, `short_description` (lead, up to ~1.5K chars), `category`, `authors`, `date`, `article_link`, `label` |
| **License** | CC0 1.0 (public domain) |

The `short_description` field is the editorial lead -- useful as a validation dataset
since it's small enough to process quickly and the leads are where metaphor density peaks.

```bash
kaggle datasets download -d irakozekelly/financial-news-sentiment-dataset-20122022 --unzip -p data/raw/huffpost/
```

### 3.3 Financial-News-Multisource (selected subsets)

**Status: To do** (streaming individual subsets to avoid 21.4 GB full download)

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/Brianferrell787/financial-news-multisource |
| **Coverage** | 1990--2025, 57.1M rows from 24 sources |
| **License** | Research use only |

Full download is 21.4 GB, but individual subsets can be streamed and filtered.
Only subsets with **full article text** are worth collecting:

| Subset | Content | Coverage | Full text? |
|--------|---------|----------|------------|
| `bloomberg_reuters` | Premium financial journalism | 2006--2013 | Yes |
| `all_the_news_2` | Broad news articles | Various | Yes |
| `reddit_finance_sp500` | Reddit posts with selftext | 2008--2025 | Yes |
| `nyt_articles_2000_present` | NYT headlines + abstracts | 2000+ | Abstract only |

```python
from datasets import load_dataset
ds = load_dataset("Brianferrell787/financial-news-multisource",
                   data_files="data/bloomberg_reuters/*.parquet",
                   split="train", streaming=True)
```

### 3.4 FNSPID

**Status: Skipped** (unusable -- all text/summary columns are null)

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/Zihan1004/FNSPID |
| **Size** | 29.6 GB |
| **Problem** | Only metadata columns (date, title, symbol, url, publisher) are populated. All text, summary, and sentiment columns are null/empty. Dataset viewer shows generation errors. |

---

## 4. Op-Ed, Opinion & Editorial Content

Opinion, editorial, and analysis text is where metaphor density is highest --
writers and speakers use vivid framing to persuade ("headwinds", "blood in the
streets", "soft landing", "war chest"). This section covers datasets explicitly
containing opinion-grade financial text.

### 4.1 Financial PhraseBank (academic benchmark)

**Status: To do**

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/takala/financial_phrasebank |
| **Size** | ~2 MB, 4,840 sentences |
| **Format** | Annotated sentences from financial news, classified by domain experts |
| **License** | Research use |

Standard benchmark for financial sentiment. Small but every sentence was selected
for its opinion content -- ideal for validating our metaphor detector on editorial text.

```python
from datasets import load_dataset
ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
```

### 4.2 FiQA (Financial Opinion Mining)

**Status: To do**

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/vibrantlabsai/fiqa |
| **Size** | ~50 MB |
| **Format** | Financial opinion text with aspect-level sentiment annotations + QA pairs |
| **License** | Research use (IEEE DataPort) |

Explicitly labeled opinion text about financial entities -- posts and headlines
with fine-grained sentiment on specific aspects (e.g. "bullish on revenue but
bearish on margins").

```python
from datasets import load_dataset
ds = load_dataset("vibrantlabsai/fiqa")
```

### 4.3 Twitter/X Financial Sentiment

**Status: To do**

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment |
| **Size** | ~10 MB |
| **Format** | Tweets about financial topics with sentiment labels |
| **License** | Check source |

Retail investor narrative on social media. Informal language with colorful
metaphors ("to the moon", "diamond hands", "bears in shambles").

```python
from datasets import load_dataset
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
```

### 4.4 Reddit Finance (r/wallstreetbets, r/investing)

**Status: To do**

| | |
|---|---|
| **Source** | https://www.kaggle.com/datasets/leukipp/reddit-finance-data |
| **Size** | ~500 MB -- 1 GB |
| **Coverage** | 2008--2025 |
| **Format** | Posts with title + selftext + engagement metrics |
| **License** | Public |

Pure retail opinion narrative. r/wallstreetbets in particular uses extremely
colorful metaphorical language during market events (2021 GME saga, COVID crash).

```bash
kaggle datasets download -d leukipp/reddit-finance-data --unzip -p data/raw/reddit_finance/
```

### 4.5 Strux Earnings Call Transcripts

**Status: To do**

| | |
|---|---|
| **Source** | https://struxdata.github.io/ |
| **Size** | ~1--2 GB, 11,950 transcripts |
| **Coverage** | 2017--2024, NASDAQ 500 + S&P 500 |
| **Format** | Transcripts (prepared remarks + Q&A) |
| **License** | Check source |

Management narrative during earnings calls is extraordinarily rich in metaphor.
CEOs and CFOs use metaphors like "headwinds", "tailwinds", "runway", "bridge",
"war chest" constantly. This is opinion-grade text from corporate insiders about
their own company's trajectory.

### 4.6 EDGAR-CORPUS (MD&A Sections)

**Status: To do** (deferred -- stream + filter for MD&A only)

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/eloukas/edgar-corpus |
| **Coverage** | 1993--2020 annual reports |
| **Format** | Pre-parsed SEC filings |
| **License** | Public (SEC data) |

Management Discussion & Analysis (MD&A) sections are the opinion-rich part of
annual reports -- where management explains "why" things happened using extensive
metaphorical framing.

### 4.7 GDELT (Global Database of Events, Language and Tone)

**Status: To do** -- high potential for Kindleberger phase mapping

| | |
|---|---|
| **Source** | https://www.gdeltproject.org/ |
| **Coverage** | 1979--present, 88M+ articles/year from 150K+ news outlets |
| **Format** | Structured metadata + tone scores (not full text by default) |
| **Access** | Free via Google BigQuery (1 TB/month free tier), or direct file downloads from AWS |
| **License** | Open |

**Why GDELT is uniquely valuable for Kindleberger analysis:**

GDELT doesn't provide full article text, but it pre-computes **tone scores** (via
VADER sentiment) for every article it indexes. For Kindleberger phase detection,
aggregate tone time series may be *more directly useful* than raw text:

- **Euphoria phase** = consistently high tone across financial articles
- **Distress/revulsion** = sharp tone collapse + high tone dispersion
- Daily/weekly aggregates of mean tone, tone dispersion (std dev), and article
  volume create a direct sentiment proxy for crisis phase identification

**Key capabilities:**
- GKG (Global Knowledge Graph) has 2,300+ theme codes -- filter for economic/financial themes
- CAMEO event codes 100-199 cover economic/financial events
- Quotes are extracted and attributed per article
- 45+ years of coverage enables long-run Kindleberger cycle analysis across multiple crises
- Academic validation: tone metrics proven predictive for bond markets, stock returns, and volatilities

**Full text reconstruction:** The `gdeltnews` Python package can reconstruct full
article text from GDELT's Web News NGrams 3.0 dataset by merging overlapping n-grams.
This means we could potentially get full text for a sample of articles during crisis
periods and run our metaphor detector on those.

**Recommended approach:**
1. Query BigQuery for financial-themed GKG records across our crisis periods
   (2008 GFC, 2011 EU debt, 2015 China, 2016 Brexit, 2018 rate hike, 2020 COVID)
2. Build tone time series (mean, dispersion, volume) per crisis window
3. Optionally reconstruct full text via `gdeltnews` for a sample during peak crisis moments

**References:**
- GKG theme codes: https://github.com/CatoMinor/GDELT-GKG-Themes
- Full text reconstruction: https://www.mdpi.com/2504-2289/10/2/45
- Chinese market bubble analysis with GDELT tone: https://www.sciencedirect.com/science/article/abs/pii/S0927538X22001056
- Narrative emotions and market crises: https://www.tandfonline.com/doi/full/10.1080/15427560.2024.2365723

### 4.8 Common Crawl News (deferred)

| | |
|---|---|
| **Source** | https://commoncrawl.org/news-crawl/ |
| **Format** | WARC files requiring processing |

Customizable but requires WARC extraction infrastructure. Deferred.

---

## 5. Hungarian Central Bank (MNB) -- Future Work

No pre-built NLP dataset exists. All publications are publicly available as PDFs on
mnb.hu and would require a custom scraper + PDF text extraction pipeline
(e.g. `pymupdf` or `pdfplumber`).

| Document type | Source | Format | Frequency | Coverage |
|---------------|--------|--------|-----------|----------|
| Meeting minutes (roviditett jegyzokonyvek) | https://www.mnb.hu/monetaris-politika/a-monetaris-tanacs/kamatmeghatarozo-ulesek-roviditett-jegyzokonyvei | PDF | ~6--8/year | 2023+ |
| Inflation reports (Inflacios jelentes) | https://www.mnb.hu/kiadvanyok/jelentesek/inflacios-jelentes | PDF | Quarterly | 2011+ |
| Financial stability reports | https://www.mnb.hu/penzugyi-stabilitas/publikaciok-tanulmanyok/penzugyi-stabilitasi-jelentesek | PDF | Twice yearly | Historical |
| Press releases | https://www.mnb.hu/en/monetary-policy/the-monetary-council/press-releases | HTML | Per meeting | Historical |
| Speeches | https://www.mnb.hu/en/pressroom/speeches-performances | HTML | Ongoing | Historical |

### Tools & references

- **HuSpaCy** (https://github.com/huspacy/huspacy) -- industrial-strength Hungarian NLP: tokenization, lemmatization, POS tagging, NER
- **awesome-hungarian-nlp** (https://github.com/oroszgy/awesome-hungarian-nlp) -- curated tool/dataset list
- **Kocsis & Matrai-Pitz (2025)** "Which Text Method to Choose for Analysing Central Bank Communication?" Financial and Economic Review, 24(4), 34--64. BERT-type models outperform GPT for central bank text. https://ideas.repec.org/a/mnb/finrev/v24y2025i4p34-64.html
