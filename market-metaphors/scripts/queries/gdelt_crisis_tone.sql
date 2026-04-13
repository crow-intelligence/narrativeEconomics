-- GDELT GKG: Daily tone aggregates for financial/economic themes
-- during Kindleberger crisis windows.
--
-- Run in Google BigQuery console: https://console.cloud.google.com/bigquery
-- Dataset: gdelt-bq.gdeltv2.gkg_partitioned
--
-- Output columns:
--   date          - publication date (YYYY-MM-DD)
--   crisis        - which crisis window this falls in
--   article_count - number of financial articles that day
--   tone_mean     - average tone (positive = optimistic, negative = pessimistic)
--   tone_std      - tone dispersion (high = mixed sentiment)
--   tone_min      - most negative article tone
--   tone_max      - most positive article tone
--
-- The tone field in GDELT GKG V2FieldTone is:
--   Tone, PositiveScore, NegativeScore, Polarity, ActivityRefDensity, SelfGroupRefDensity
-- We extract the first value (overall tone) which ranges roughly -10 to +10.

WITH crisis_windows AS (
  SELECT 'gfc_2008' AS crisis, '2007-06-01' AS start_date, '2009-06-30' AS end_date
  UNION ALL SELECT 'eu_debt_2011', '2010-05-01', '2011-09-30'
  UNION ALL SELECT 'china_2015', '2015-06-01', '2015-09-30'
  UNION ALL SELECT 'brexit_2016', '2016-02-01', '2016-06-30'
  UNION ALL SELECT 'rate_hike_2018', '2018-01-01', '2018-12-31'
  UNION ALL SELECT 'covid_2020', '2020-01-01', '2020-06-30'
),

financial_articles AS (
  SELECT
    PARSE_DATE('%Y%m%d', CAST(DATE AS STRING)) AS pub_date,
    -- Extract overall tone (first semicolon-delimited value from V2Tone)
    SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    DocumentIdentifier AS url
  FROM
    `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE
    -- Date filter: cover all crisis windows (2007-2020)
    _PARTITIONTIME >= TIMESTAMP('2007-06-01')
    AND _PARTITIONTIME <= TIMESTAMP('2020-06-30')
    -- Financial/economic theme filter
    -- V2Themes contains semicolon-separated theme codes
    AND (
      V2Themes LIKE '%ECON_%'
      OR V2Themes LIKE '%TAX_%'
      OR V2Themes LIKE '%WB_2331_STOCK_MARKET%'
      OR V2Themes LIKE '%WB_2898_FINANCIAL_SECTOR%'
      OR V2Themes LIKE '%WB_621_DEBT%'
      OR V2Themes LIKE '%WB_696_PUBLIC_FINANCE%'
      OR V2Themes LIKE '%WB_2025_MONETARY_POLICY%'
      OR V2Themes LIKE '%WB_2026_INFLATION%'
      OR V2Themes LIKE '%WB_840_BANKING%'
      OR V2Themes LIKE '%MANMADE_DISASTER_ECONOMIC_CRISIS%'
      OR V2Themes LIKE '%CRISISLEX_%'
      OR V2Themes LIKE '%EPU_ECONOMY%'
      OR V2Themes LIKE '%EPU_POLICY%'
    )
    -- English language only
    AND TranslationInfo = ''
)

SELECT
  fa.pub_date AS date,
  cw.crisis,
  COUNT(*) AS article_count,
  ROUND(AVG(fa.tone), 4) AS tone_mean,
  ROUND(STDDEV(fa.tone), 4) AS tone_std,
  ROUND(MIN(fa.tone), 4) AS tone_min,
  ROUND(MAX(fa.tone), 4) AS tone_max
FROM
  financial_articles fa
JOIN
  crisis_windows cw
  ON fa.pub_date >= PARSE_DATE('%Y-%m-%d', cw.start_date)
  AND fa.pub_date <= PARSE_DATE('%Y-%m-%d', cw.end_date)
GROUP BY
  fa.pub_date, cw.crisis
ORDER BY
  cw.crisis, fa.pub_date
