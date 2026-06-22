## Overview
Exploratory Data Analysis (EDA) on a 1 million row YouTube recommendation 
dataset, covering data cleaning, statistical analysis, visualization, 
and an interactive Tableau dashboard.

## Dataset
- **Source:** youtube recommendation dataset.csv
- **Size:** 1,000,000 rows × 14 columns
- **Key columns:** user_id, video_id, video_duration, watch_time, liked, 
  commented, subscribed_after, category, device, watch_time_of_day, 
  recommended, clicked, timestamp, watch_percent

## Tools Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, SciPy)
- **Jupyter Notebook** — EDA and cleaning
- **Tableau** — Interactive dashboard

## Steps Performed
### 1. Data Loading
Loaded CSV with 1,000,000 rows and 14 columns.

### 2. Data Cleaning
Found and fixed 5 real issues:
- `liked` column had mixed values (yes/no/2/blank) — standardized to 0/1
- 1,000 rows had `video_duration` stuck at exactly 100,000 (corrupted batch) — dropped
- 1,000 rows had `video_duration = 0` (division by zero risk) — dropped
- 2,000 rows had mixed timestamp formats — coerced and dropped unparsable
- `category` had inconsistent casing and typos — normalized
- **Final clean dataset: 998,000 rows**

### 3. Descriptive Statistics
- Average watch percent: **75.0%**
- Click rate: **30.1%**
- Like rate: **30.1%**
- Comment rate: **10.0%**
- Subscribe rate: **5.0%**

### 4. Univariate Analysis
- Histogram of watch percent distribution
- Box plot of video duration by category
- Bar chart of views per category

### 5. Bivariate Analysis
- Scatter plot: video duration vs watch percent
- Correlation heatmap across engagement variables
- Recommended vs organic engagement comparison

### 6. Data Visualization
- Bar, line, and pie charts across key dimensions

### 7. Insights
1. Short-form content generates over twice the proportion of highly 
   engaged users compared to long-form content
2. Viewer engagement consistently declines as video length increases
3. Viewer engagement is slightly higher during evenings and nights
4. A/B test (p-value: 0.1347 > 0.05) confirms recommended videos do NOT 
   significantly outperform organic views — the algorithm shows no 
   measurable impact on engagement
5. Near-zero correlations across all variables suggest this dataset 
   may be synthetically generated

### 8. Bonus Features
- Feature Engineering (engagement score, highly engaged flag, duration buckets)
- Time Series analysis of monthly view trends
- A/B Testing (recommended vs organic click rate)

## Dashboard
Interactive Tableau dashboard with 8 charts and cross-filtering by 
category, time of day, and recommended status.
