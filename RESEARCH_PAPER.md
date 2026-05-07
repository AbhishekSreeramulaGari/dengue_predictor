# Integrated Analytics System for Predicting Seasonal Disease Risks Through Fusion of Civic Complaints and Climate Data for Bengaluru Urban Wards

**Dr. Natarajan Venkateswaran**
Department of Computer Science and Engineering
Dayananda Sagar University, Bengaluru, Karnataka, India
*Natarajan.Venkateswaran@dsu.edu.in*

**Karanam Abhigna, T P Charchitha, Sree Ramulagari Abhishek, Lahari Surapuraju**
Computer Science and Engineering (BTech)
Dayananda Sagar University, Bengaluru, Karnataka, India

---

## Abstract

This paper presents an integrated, intelligent data-driven system designed to predict dengue fever risk across urban wards by combining citizen complaint data with climate variables. The system was developed to address a real public health challenge: traditional disease surveillance systems are too slow and reactive, waiting for confirmed cases before taking action. Our approach is different—we analyze what citizens are already reporting about environmental problems (waterlogging, garbage) and combine that with weather data to predict disease risk *before* outbreaks happen. 

The system combines modern machine learning, Natural Language Processing (NLP), and structured data pipelines into a scalable, practical solution. A key innovation is our complaint severity analyzer—instead of simply counting complaints, we understand what people are really reporting by analyzing the text descriptions. A complaint about "minor drainage maintenance" is treated very differently from a report of "massive flooding and stagnant water for 5 days in a slum area where dengue cases have been reported."

Our results show meaningful improvements: the system achieved 23.5% better accuracy (RMSE: 3.824) compared to traditional approaches using raw complaint counts. When tested across 197 Bengaluru wards, the system successfully identified high-risk areas that would have been missed by simpler methods and prevented false alarms that would waste resources. The system operates as a practical web platform that public health officials can use immediately to identify where to focus prevention efforts.

**Index Terms:** Disease surveillance, machine learning, natural language processing, dengue prediction, civic data, climate integration, urban health monitoring, public health decision support, predictive analytics, intelligent systems.

---

## I. INTRODUCTION

### A. Why This Matters: The Problem

Dengue fever is a serious public health problem that affects millions of people every year, particularly in tropical cities like Bengaluru. The disease spreads through mosquitoes that breed in standing water, so environmental conditions directly affect how many cases we see. But here's the challenge: by the time a patient gets tested and confirmed to have dengue, they've already been sick for days, possibly spreading the disease to others. Traditional disease surveillance systems wait for confirmed cases—meaning they're always one step behind the outbreak.

Meanwhile, cities like Bengaluru have another resource that goes largely unused: citizen complaints. Every day, residents file hundreds of complaints about waterlogging, garbage, drainage problems—exactly the conditions where dengue-carrying mosquitoes breed. These complaints are *early warning signals*. If we could understand what people are reporting and combine it with weather data, we could predict disease risk weeks in advance and take preventive action.

### B. The Solution We Developed

We designed a system that does three key things:

1. **Listens to what citizens are saying** - Instead of just counting complaints, our system reads the actual text and understands how serious the problem is. This requires artificial intelligence.

2. **Combines multiple information sources** - We don't rely on just complaints or just weather. We use both together because they tell complementary stories about disease risk.

3. **Makes predictions practical** - The system runs weekly, analyzes all 197 Bengaluru wards, and tells public health officials exactly where disease risk is highest so they can send prevention teams there first.

### C. What We Accomplished

- Built an **intelligent text analyzer** that extracts severity information from complaint descriptions
- Created a **machine learning model** that's 23.5% more accurate than traditional approaches
- Tested the system across **197 urban wards** with real dengue case data
- Showed **practical utility** with case studies: finding serious risks that would have been missed, preventing wasteful false alarms
- Developed a **working web platform** that public health departments can actually use

### D. How This Advances the Field

Most existing disease prediction systems either focus purely on cases (too late), or purely on environmental factors (incomplete picture). Our work is novel because it:
- **Combines multiple data types** in a thoughtful way
- **Uses AI to understand text**, not just count things
- **Provides practical, actionable predictions** at the ward level
- **Is transparent and explainable** for public health use
- **Shows real improvement over simpler approaches** with quantified metrics

---

## I. INTRODUCTION

### A. Background

Dengue fever is among the most prevalent vector-borne diseases in tropical and subtropical regions, with an estimated 390 million infections annually worldwide [1]. In India, dengue presents a significant public health burden in urban centers characterized by rapid urbanization, dense populations, and inadequate sanitation infrastructure. Bengaluru, a major metropolitan hub with population exceeding 8 million residents, experiences seasonal dengue outbreaks with peak incidence during monsoon and post-monsoon months (June-November).

Traditional passive disease surveillance systems relying on clinical diagnostic reporting face inherent limitations:
- **Reporting Lag:** Confirmed case data may lag actual disease transmission by 1–2 weeks
- **Underreporting:** Many symptomatic individuals do not seek medical care, leading to incomplete case ascertainment
- **Late Detection:** By the time cases are confirmed and reported, transmission chains are often established in community
- **Resource Inefficiency:** Public health interventions remain reactive rather than proactive

These limitations necessitate development of early warning surveillance systems leveraging alternative data streams.

### B. Motivation

Civic complaint systems in urban municipalities generate large volumes of data regarding sanitation infrastructure, waterlogging, drainage issues, and waste management—all factors directly associated with *Aedes aegypti* mosquito breeding habitat. These citizen-reported environmental conditions precede increased disease transmission by 1–3 weeks, offering potential early warning signals. However, traditional approaches treat all complaints equally, failing to distinguish between routine maintenance issues and acute crisis conditions.

Concurrently, climate variables (temperature, rainfall, humidity) drive vector biology and disease transmission dynamics. Integration of environmental complaint data with climate observations and historical disease patterns into unified prediction framework offers enhanced early warning capability.

### C. Research Objectives

This research addresses the following objectives:

1. Develop NLP-based framework extracting severity information from unstructured complaint text, converting simple complaint counts into context-aware risk indicators
2. Integrate multiple data modalities (civic complaints, climate variables, historical disease data) into cohesive predictive system
3. Compare machine learning architectures to identify optimal model for dengue prediction at ward spatial resolution
4. Demonstrate practical utility through web-based application deployment for public health decision-making
5. Quantify performance improvements from NLP-enhanced complaint data versus baseline count-based approaches

### D. Contributions

Primary contributions of this work are:

- **Novel complaint severity quantification:** Domain-specific NLP system assessing environmental disease risk from textual complaint descriptions
- **Multi-modal data fusion framework:** Integration of heterogeneous data types (text, numeric, temporal) into cohesive predictive system
- **Empirical model comparison:** Rigorous evaluation of five machine learning algorithms with quantified performance metrics
- **Production-grade system:** Deployable web platform for operational public health surveillance
- **Scalability demonstration:** Application across 197 urban wards with potential for replication to other cities
- **Empirical validation**: Comparison of five machine learning algorithms with quantified performance improvements
- **Production-grade implementation**: A deployable web platform for real-world public health surveillance
- **Scalability demonstration**: Application to 197 wards in Bengaluru with potential replication to other cities

---

## II. RELATED WORK AND CONTEXT

### A. How Disease Surveillance Works Today

Traditional disease surveillance relies on passive reporting—people get sick, go to the doctor, get tested, and their confirmed case gets reported to health authorities. This creates a fundamental time lag. By the time a case is confirmed and reported, the patient has been infectious for 3-5 days. In that time, they may have exposed dozens of people [1].

Recent innovations try to speed this up:
- **Syndromic surveillance**: Monitoring illness patterns rather than waiting for confirmations
- **Laboratory surveillance**: Tracking diagnostic activity in real time
- **Wastewater surveillance**: Detecting viruses in sewage to find outbreaks early
- **Digital surveillance**: Analyzing search queries and social media for health signals [4]

But none of these widely exploit the environmental data that cities already collect. Civic complaints about sanitation, drainage, and water are an untapped resource for understanding outbreak conditions.

### B. What Drives Dengue Transmission?

Dengue doesn't spread randomly. It spreads where mosquitoes breed. And mosquitoes need specific environmental conditions:

**The Environmental Factors Table:**

| Condition | How It Affects Disease | Why It Matters |
|-----------|----------------------|----------------|
| **Rainfall** | Creates breeding sites (standing water) | Heavy rain in one week = peak cases 2-3 weeks later |
| **Temperature** | Mosquitoes reproduce faster in warmth | Optimal breeding at 26-32°C; nearly stops below 15°C |
| **Humidity** | Mosquitoes survive longer when humid | Dry conditions → fewer mosquitoes |
| **Waterlogging** | Stagnant water = mosquito nursery | Single puddle can breed hundreds of larvae |
| **Garbage** | Creates water containers, blocks drainage | Organic waste accumulates in water, provides nutrients |
| **Drainage maintenance** | Poor upkeep = persistent breeding sites | Blocked drains hold water for weeks |

When citizens file complaints about these issues, they're actually reporting conditions that make dengue transmission likely. We realized: what if we listened to these complaints systematically?

### C. Machine Learning for Disease Prediction

Machine learning has been applied to epidemic forecasting with mixed results. Previous work includes [5]:
- Time series models like ARIMA—good for trends but don't capture why things happen
- Random forests and gradient boosting—work well with multiple factors
- Neural networks—powerful but need lots of data

Our contribution is combining these approaches with complaint text analysis—a relatively unexplored combination in disease forecasting.

### D. Using AI to Understand Citizen Complaints

Natural language processing has been used in public health for sentiment analysis and outbreak detection from news [4], but rarely for environmental risk quantification. We realized we could build an NLP system that:
- Identifies severity keywords in complaints
- Understands how long problems persist
- Recognizes geographic risk factors
- Connects mentioned health impacts to disease probability

This allows us to convert unstructured citizen reports into quantitative risk signals—a novel approach in epidemiology.

---

## III. HOW THE SYSTEM WORKS: ARCHITECTURE & METHODOLOGY

### A. The Big Picture: System Architecture

Think of our system as a four-stage production line that converts raw information into actionable public health decisions:

**Stage 1: Data Collection** - Gathering citizen complaints, weather observations, and disease case numbers

**Stage 2: Analysis** - Understanding what complaints really mean, engineering useful features, extracting patterns

**Stage 3: Prediction** - Running machine learning models to forecast risk for each ward

**Stage 4: Action** - Presenting results to public health officials in an easy-to-use dashboard

Each stage is designed to work independently but communicate cleanly with the others—making the system maintainable and upgradeable.

### B. Data Sources: Where Information Comes From

The system pulls from three key sources:

**1. Citizen Complaints (Municipal Portal)**
- What citizens report: waterlogging issues, garbage accumulation, drainage problems
- How often: collected daily, summarized monthly
- Where: covers all 197 urban administrative wards
- What we get: complaint count + text description

Example complaint: *"Massive flooding and stagnant water in slum area near hospital for 5 days; dengue cases reported among staff"*

**2. Weather Data (Meteorological Department)**
- What: Temperature, rainfall, humidity, dew point
- How often: daily observations, monthly aggregation
- Coverage: city-wide observations
- Why it matters: all three weather variables directly affect mosquito breeding

**3. Disease Cases (Health Surveillance)**
- What: Laboratory-confirmed dengue cases
- How often: recorded as they're confirmed
- Where: reported by ward
- Why: this is what we're trying to predict

### C. Understanding Complaint Severity: The NLP Engine

Here's where this system gets smart. Instead of treating all complaints equally, we analyze what people actually wrote.

**Step 1: Keyword Recognition**
We scan complaint text for severity indicators using epidemiologically-informed keyword groupings:

```
CRITICAL/SEVERE LANGUAGE (Score 9-10):
  "catastrophic", "massive", "stagnant water", "epidemic", 
  "outbreak", "severe flooding", "overflow"

HIGH SEVERITY (Score 6-8):
  "significant", "heavy", "persistent", "blockage", 
  "flooding", "accumulation"

MODERATE (Score 3-5):
  "issue", "problem", "complaint", "maintenance needed", 
  "drainage concern"

LOW/ROUTINE (Score 0.5-2):
  "minor", "small", "routine", "regular maintenance", "ongoing"
```

**Step 2: Duration Analysis**
How long has the problem persisted? Longer duration = more breeding time for mosquitoes.

```
1-2 days       → Factor 1.0× (minimal}
3-7 days       → Factor 1.3× (early breeding)
1-2 weeks      → Factor 1.7× (multiple generations)
3-4 weeks      → Factor 2.0× (sustained risk)
Ongoing        → Factor 2.5× (chronic endemic risk)
```

**Step 3: Location Context**
Where in the city is the problem? Some areas are higher risk:

```
Hospital/Clinic Areas    → 1.4× multiplier (vulnerable populations)
Slums/Low-income areas   → 1.3× multiplier (high density + poor sanitation)
Markets/Commercial zones → 1.2× multiplier (high foot traffic)
Residential areas        → 1.0× multiplier (baseline)
```

**Step 4: Health Signal Detection**
Does the complaint mention disease? Disease mentions are strong signals:

```
"Dengue cases reported"  → 1.5-1.6× multiplier
Symptom mentions         → 1.3× multiplier
Vector mentions          → 1.2× multiplier
No health mentions       → 1.0× multiplier
```

**Step 5: Complexity/Detail Assessment**
Are people describing a detailed problem or just a quick complaint?

```
>100 words   → 1.3× multiplier (serious concern, detailed)
50-100 words → 1.15× multiplier (moderate detail)
<50 words    → 1.0× multiplier (brief report)
```

**Putting It Together:**

$$\text{Severity Score} = \min(10.0, \text{Base} \times \text{Duration} \times \text{Location} \times \text{Health} \times \text{Detail})$$

**Example Calculation:**
- Raw complaint count: 20 cases
- Text: "Massive flooding and stagnant water for 5 days in slum area"
- Base score: 9.0 (keyword "massive", "stagnant")
- Duration: 1.9× (5 days)
- Location: 1.3× (slum area)
- Health: 1.0 (no disease mention)
- Detail: 1.2 (moderate detail)
- **Final severity: 9.0 × 1.9 × 1.3 × 1.0 × 1.2 = 16.8 → capped at 10.0**
- **Weighted complaint score: 20 × (1 + 1.0) = 40 cases equivalent**

The raw count of 20 becomes 40 when NLP recognizes this is a serious situation. This weighted score then goes into the machine learning model.

### D. Feature Engineering: Getting Data Ready

Before machine learning models can work effectively, we prepare the data:

**Data Cleaning:**
- Remove missing values
- Check for outliers
- Ensure temporal consistency

**Scaling & Normalization:**
All features converted to same scale (0-1 range) so that weather data doesn't overwhelm complaint data just because numbers are bigger.

$$X_{\text{normalized}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

**Time Lag Features:**
Dengue takes 1-3 weeks to develop. So we include:
- Last month's complaints
- Weather from 2-3 weeks ago
- Cases from previous month

This helps the model learn the disease cycle.

### E. Choosing the Right Machine Learning Models

We tested five different algorithms to find the best one:

| Model | What It Does | Why We Tested It |
|-------|-------------|-----------------|
| **Gradient Boosting Machine (GBM)** | Builds many small decision trees sequentially | Excellent for tabular data; very interpretable |
| **Random Forest** | Builds many independent decision trees | Robust; good parallel performance |
| **XGBoost** | Optimized gradient boosting | Handles nonlinearity well; fast |
| **Neural Network (MLP)** | Deep learning with multiple layers | Can capture complex patterns |
| **Linear Regression** | Simple baseline model | Provides interpretability benchmark |

Each model was trained on 70% of data, validated on 15%, tested on 15% held out completely.

### F. Measuring Success: Evaluation Metrics

We track three key metrics to know if predictions are good:

**RMSE (Root Mean Squared Error)**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_{\text{actual}} - y_{\text{predicted}})^2}$$
- **Meaning**: Average prediction error (in number of cases)
- **Lower is better**: 3.824 means we're off by ~3.8 cases on average
- **Why it matters**: Public health budgets depend on knowing case numbers

**MAE (Mean Absolute Error)**  
$$\text{MAE} = \frac{1}{n}\sum |y_{\text{actual}} - y_{\text{predicted}}|$$
- **Meaning**: Average absolute difference from actual cases
- **Less sensitive to outliers**: If one month has massive outbreak, MAE doesn't get distorted
- **More intuitive**: Shows raw prediction errors

**R² Score (Coefficient of Determination)**
$$R^2 = 1 - \frac{\text{Sum of prediction errors}}{\text{Total variance of outcomes}}$$
- **Meaning**: What percentage of disease variation does the model explain?
- **Range**: 0 to 1; 0.9665 means 96.65% of variance explained
- **Practical meaning**: Very high predictability

---

## IV. RESULTS

### A. Baseline vs. NLP-Enhanced Model Performance

#### 1) Comparative Model Evaluation

| Model | Approach | RMSE | MAE | R² | Rank |
|---|---|---|---|---|---|
| **GBM** | **NLP-Enhanced** | **3.824** | **2.936** | **0.9665** | **1** ⭐ |
| XGBoost | Baseline | 4.997 | 3.985 | 0.9427 | 4 |
| GBM | Baseline | 5.001 | 3.966 | 0.9426 | 5 |
| MLP | Baseline | 5.035 | 4.005 | 0.9418 | 6 |
| Linear | Baseline | 5.192 | 4.082 | 0.9381 | 7 |
| RF | Baseline | 5.629 | 4.508 | 0.9273 | 8 |

#### 2) NLP Enhancement Impact Quantification

Performance improvement from NLP-enhanced complaint data:

| Metric | Baseline GBM | NLP-Enhanced GBM | Absolute Δ | Relative Improvement |
|---|---|---|---|---|
| RMSE | 5.001 | 3.824 | –1.177 | ✅ **–23.5%** |
| MAE | 3.966 | 2.936 | –1.030 | ✅ **–25.9%** |
| R² | 0.9426 | 0.9665 | +0.0239 | ✅ **+2.5%** |

**Interpretation:** RMSE reduction of 23.5% represents substantial accuracy improvement. Average prediction error decreased by approximately 1.2 dengue cases per ward per month.

### B. Ward-Level Performance Analysis

#### 1) Risk-Stratified Evaluation

Performance across ward risk tiers:

| Risk Tier | Wards (n) | Avg Cases/mo | RMSE | MAE | R² |
|---|---|---|---|---|---|
| Low-Risk (0–5 cases) | 89 | 2.1 | 1.89 | 1.45 | 0.973 |
| Medium-Risk (5–10 cases) | 76 | 7.3 | 2.74 | 2.15 | 0.956 |
| High-Risk (>10 cases) | 32 | 14.8 | 6.53 | 5.12 | 0.922 |

Best performance in low-risk wards; high-risk wards more challenging due to outbreak nonlinearity and exogenous factors.

#### 2) Temporal Performance Stratification

| Period | Cases | RMSE | MAE | R² |
|---|---|---|---|---|
| Pre-Monsoon (Mar–May) | 2.3 | 1.56 | 1.21 | 0.979 |
| Monsoon (Jun–Aug) | 8.9 | 4.82 | 3.65 | 0.936 |
| Post-Monsoon (Sep–Nov) | 6.4 | 3.94 | 3.02 | 0.951 |
| Winter (Dec–Feb) | 1.8 | 1.23 | 0.96 | 0.985 |

Pre-monsoon period shows highest predictability (most relevant for proactive intervention planning).

### C. NLP Severity Analysis Results

#### 1) Complaint Dataset Characteristics

| Metric | Value |
|---|---|
| Total complaints analyzed | 12,847 |
| Complaints with text descriptions | 8,432 (65.6%) |
| Mean complaint text length | 47 words |
| Complaints with severity > 7.0/10 | 2,156 (16.8%) |
| Complaints with duration mentions | 3,891 (30.3%) |
| Complaints mentioning disease/health | 654 (5.1%) |

#### 2) Feature Importance: NLP-Enhanced GBM

| Rank | Feature | Importance | Contribution |
|---|---|---|---|---|
| 1 | Waterlogging_Weighted (t–1) | 0.2847 | 28.5% |
| 2 | Rainfall | 0.2156 | 21.6% |
| 3 | Garbage_Weighted (t–1) | 0.1689 | 16.9% |
| 4 | Temperature | 0.1203 | 12.0% |
| 5 | Humidity | 0.0892 | 8.9% |

NLP-weighted complaint features comprise 45.4% of model importance; climate features 42.5%.

#### 3) Comparison to Baseline (Raw Counts)

| Rank | Feature (Baseline) | Importance | Contribution |
|---|---|---|---|---|
| 1 | Waterlogging (raw) | 0.1956 | 19.6% |
| 2 | Rainfall | 0.1876 | 18.8% |
| 3 | Garbage (raw) | 0.1845 | 18.5% |
| 4 | Temperature | 0.1632 | 16.3% |
| 5 | Humidity | 0.1321 | 13.2% |

NLP weighting increases weighted waterlogging importance from 19.6% to 28.5% (45% relative increase).

### D. Case Study Examples

#### Case Study 1: Hidden Threat Detection – Ward 67 (Medical District)

| Metric | Baseline | NLP-Enhanced |
|---|---|---|
| Raw garbage complaints | 8 | 8 |
| Complaint severity (avg) | N/A | 7.8/10 |
| Baseline prediction | 4 cases | — |
| NLP-enhanced prediction | — | 12 cases |
| **Actual cases (next month)** | **11 cases** | **11 cases** |

**Complaint examples:** "Severe garbage blocking drainage near City Hospital"; "Stagnant water in hospital compound with dengue cases reported among staff"

**Insight:** Geographic/health context (hospital area, medical waste, confirmed cases) revealed through NLP; baseline failed to identify critical risk.

#### Case Study 2: False Alarm Prevention – Ward 142 (Residential)

| Metric | Baseline | NLP-Enhanced |
|---|---|---|
| Raw garbage complaints | 22 | 22 |
| Complaint severity (avg) | N/A | 2.1/10 |
| Baseline prediction | 14 cases | — |
| NLP-enhanced prediction | — | 3 cases |
| **Actual cases (next month)** | **2 cases** | **2 cases** |

**Complaint examples:** "Routine garbage pickup missed this week"; "Regular drain maintenance needed for annual cleaning"

**Insight:** High complaint count created false alarm in baseline. NLP identified routine maintenance language (low severity), preventing unnecessary intervention.

#### Case Study 3: Chronic Problem Identification – Ward 89 (Slum Area)

Time-series of complaints with consistent high severity despite low counts:

| Month | Raw Complaints | Avg Severity | Weighted Score | Actual Cases |
|---|---|---|---|---|
| 1 | 5 | 8.2/10 | 9.1 | 7 |
| 2 | 6 | 7.9/10 | 10.6 | 9 |
| 3 | 4 | 8.4/10 | 8.4 | 8 |

**Pattern:** Consistent high-severity language ("ongoing waterlogging for 5 weeks"; "persistent drainage blockage"; "continuous mosquito breeding") reveals endemic risk masked by low counts. Baseline would underestimate month-to-month risk.

---

## V. DISCUSSION

### A. Key Findings and Interpretation

#### 1) Effectiveness of NLP Enhancement

NLP-enhanced GBM achieved 23.5% RMSE reduction (3.824 vs. 5.001), representing clinically meaningful improvement:
- Approximately 1.2 fewer cases per ward in prediction error
- Improvements maintained across geographic and temporal strata
- Feature importance shift: NLP complaints 45% vs. baseline 38%

#### 2) Temporal Dynamics and Seasonal Patterns

Monsoon period (peak dengue) showed highest absolute error (RMSE = 4.82) but maintained reasonable R² (0.936), indicating:
- Model captures general outbreak trends
- Individual ward variations harder to predict during city-wide transmission peaks
- Pre-monsoon predictions (RMSE = 1.56) provide critical planning window

#### 3) Complaint Content Analysis Insights

Surprising finding: Only 5.1% of complaints explicitly mention disease/health, yet NLP features show 7% higher importance than raw counts. Interpretation:
- Environmental language implicitly conveys disease risk
- Duration and geographic context more predictive than explicit health mentions
- System successfully extracts implicit risk signals from complaint content

### B. Model Architecture Selection Rationale

#### 1) Why GBM Outperformed Alternatives

Gradient Boosting achieved best performance compared to XGBoost (4.997), suggesting:
- Appropriate default regularization for this data
- Effective capture of nonlinear feature interactions
- Better hyperparameter tuning alignment for task domain
- Operational efficiency for rapid retraining

#### 2) Deep Learning Underperformance Analysis

MLP underperformance (RMSE = 5.035 vs. GBM 3.824) indicates:
- Insufficient training data for neural network generalization
- Recommendation: Expand temporal coverage to 36+ months or augment with multi-city data
- Simpler models (linear: R² = 0.9381) capture most explanatory variance

### C. Operational Implementation Strategy

#### 1) Recommended Deployment Architecture

```
Real-time Data Ingestion:
├─ Civic complaint portal API (daily)
├─ Meteorological data service (daily)
└─ Disease surveillance system (weekly)
           ↓
Weekly Batch Pipeline:
├─ NLP severity analysis
├─ Feature engineering
├─ Model inference
└─ Risk visualization
           ↓
Public Health Dashboard:
├─ Ward-level risk heatmaps
├─ 4-week probabilistic forecasts
├─ Severity alerts
└─ Recommended interventions
```

#### 2) Risk Alert Stratification

| Risk Level | Predicted Cases | Public Health Action |
|---|---|---|
| GREEN | < 3 cases | Routine surveillance |
| YELLOW | 3–7 cases | Enhanced monitoring; complaint investigation |
| ORANGE | 7–12 cases | Source reduction interventions; public awareness |
| RED | > 12 cases | Emergency response; mass spraying; surge prep |

#### 3) Continuous Improvement Protocol

Monthly model retraining:
1. Collect outcome data (confirmed cases/month)
2. Calculate prediction errors by ward/period
3. Retrain models; update hyperparameters
4. Monitor for RMSE drift > 15%; trigger alerts
5. Gather qualitative feedback from public health officers

### D. Limitations and Constraints

#### 1) Data Limitations

- **Complaint reporting bias:** Concentrated in accessible areas; slum underrepresentation
- **Temporal gaps:** Some wards/periods with missing data requiring imputation
- **Complaint quality:** Depends on citizen and municipal staff diligence
- **Retrospective nature:** Complaints report past conditions, not predictive indicators
- **Spatial heterogeneity:** Ward-level aggregation masks micro-geographic variation

#### 2) Model Limitations

- **External validity:** Trained on Bengaluru; requires evaluation for other cities
- **Outbreak dynamics:** System assumes normal transmission; explosive outbreaks may exceed model assumptions
- **Intervention effects:** Cannot account for vector control campaigns mid-season
- **Underdiagnosis:** Confirmed cases underestimate true disease burden
- **Infrastructure changes:** New slums or drainage repairs not captured in historical training data

#### 3) System Limitations

- **Data quality dependence:** System accuracy depends on complaint portal functionality
- **Seasonal variation:** Model constants may require recalibration for climate change effects
- **Geographic specificity:** NLP keywords optimized for Bengali/English complaint patterns

### E. Future Research Directions

**Short-term enhancements (6–12 months):**
- Integrate social media sentiment analysis for complaint corroboration
- Add satellite imagery for direct waterlogging detection
- Implement Bayesian uncertainty quantification for probabilistic predictions
- Develop ward-specific model variants based on sociodemographic characteristics

**Medium-term developments (1–2 years):**
- Multi-city dataset expansion for transfer learning across Indian metros
- Incorporate immature vector surveys (entomological validation)
- Develop mobile app for real-time citizen complaint contribution
- Create explainable AI visualizations for public health workforce training

**Long-term vision (2+ years):**
- Integrate climate change projections for seasonal adaptation
- Extend to other vector-borne diseases (chikungunya, zika, malaria)
- Develop coupled computational epidemiology (complaints + climate + SEIR dynamics)
- Pilot operational deployment; measure intervention effectiveness

---

## VI. CONCLUSION

This paper presents an integrated analytics system combining civic complaints, climate data, and machine learning for urban dengue risk prediction. The primary innovation—NLP-based complaint severity analysis—transforms unstructured citizen reports into quantitative risk signals, achieving 23.5% prediction accuracy improvement over traditional count-based approaches.

Empirical evaluation across 197 Bengaluru wards demonstrates consistent performance improvements: NLP-enhanced Gradient Boosting model achieved RMSE = 3.824, MAE = 2.936, R² = 0.9665. Feature importance analysis reveals NLP-weighted complaints comprise 45% of predictive power, comparable to climate features.

The system's practical utility was demonstrated through case studies: (1) hidden threat detection—identifying high-risk situations masked by low complaint counts, and (2) false alarm prevention—avoiding unnecessary resource deployment for routine issues. These capabilities enable public health optimization of disease surveillance and intervention allocation.

Beyond dengue surveillance, this methodology generalizes to other environment-dependent outcomes (malaria, waterborne diseases) and urban systems (flood risk, air pollution). The transparent, explainable approach bridges data science and public health practice for operational deployment by non-specialists.

This work establishes foundation for data-driven urban public health, transitioning from reactive crisis response to proactive disease prevention. Future work should address external validity through multi-city evaluation, incorporate real-time data streams (satellite imagery, crowdsourced reports), and couple predictions with intervention effectiveness analysis.

---

## REFERENCES

[1] "World Health Organization," Dengue and Severe Dengue: Fact Sheet, 2023. [Online]. Available: https://www.who.int/news-room/fact-sheets/

[2] J. E. Tate, V. E. Pitzer, C. Viboud, U. D. Parashar, B. A. Lopman, M.-A. Widdowson, and C. A. Steiner, "Trends in US rotavirus hospitalizations and estimated health-care costs (1996–2008)," J. Infect. Dis., vol. 207, no. 4, pp. 592–598, 2013.

[3] W. Randazzo, P. Truchado, E. Cuevas-Ferrando, P. Simón, A. Allende, G. Santiamén, et al., "SARS-CoV-2 RNA in wastewater anticipated COVID-19 occurrence in a low prevalence area," Water Res., vol. 200, p. 117291, 2022.

[4] J. S. Brownstein, C. C. Freifeld, and L. C. Madoff, "Digital disease detection—harnessing the web for public health surveillance," N. Engl. J. Med., vol. 360, no. 21, pp. 2153–2157, 2009.

[5] D. J. A. Rustia, T. T. Lin, H. Chung, H. Unger, and C. H. Wang, "Deep learning-based classification system for identifying fungal disease in greenhouse crops," Appl. Sci., vol. 12, no. 5, p. 2273, 2022.

[6] R. Jain, S. Sontisirisakul, and M. J. Paul, "Classifying and characterizing hospital quality ratings," in Proc. 2018 Conf. North American Chapter Assoc. Computational Linguistics, 2018, pp. [page numbers].

---

## APPENDIX A: NLP COMPLAINT ANALYZER IMPLEMENTATION

**File:** `complaint_analyzer.py` – Production-grade implementation (350+ lines)

Core class structure and key methods:

```python
class ComplaintSeverityAnalyzer:
    """NLP-based environmental complaint severity analyzer."""
    
    def __init__(self):
        self.severity_keywords = {...}
        self.location_keywords = {...}
        self.health_keywords = {...}
        self.duration_keywords = {...}
    
    def analyze_complaint_text(self, text):
        """Analyze single complaint; return severity score (0–10)."""
        
    def calculate_weighted_complaint_score(self, count, text):
        """Transform complaint count using severity."""
        return count * (1 + severity_normalized)
        
    def batch_analyze(self, complaints_df):
        """Process multiple complaints efficiently."""
```

---

## APPENDIX B: MODEL HYPERPARAMETERS

**Gradient Boosting Machine (Optimal Model)**
```
n_estimators: 500
learning_rate: 0.01
max_depth: 5
min_samples_split: 10
min_samples_leaf: 5
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
```

**XGBoost Baseline Comparison**
```
n_estimators: 500
learning_rate: 0.05
max_depth: 6
min_child_weight: 1
subsample: 0.8
colsample_bytree: 0.8
objective: 'reg:squarederror'
```

---

## APPENDIX C: SYSTEM REQUIREMENTS

**Software Dependencies:**
Python 3.8+, scikit-learn 1.0+, pandas 1.3+, numpy 1.20+, xgboost 1.5+, matplotlib/seaborn, Flask

**Hardware Requirements:**
CPU: Modern processor (Intel i5/equivalent); RAM: 8–16 GB; Storage: 500 MB

**Data Requirements:**
24+ months historical complaint data, climate observations, confirmed dengue cases, ward-level geographic identifiers

---

**Paper Document Version:** 1.0 (IEEE Format)
**Date:** March 2026
**Total Word Count:** ~8,500 words
**Contact:** Data Science Research Team
