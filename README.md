# HepatoGuard AI - Post-Liver Transplant Disease & Complication Prediction

HepatoGuard AI is a clinical machine learning prognostic application designed to predict and stratify post-operative complications in liver transplant recipients using 22 multi-modal biomarkers.

---

## 🌟 Key Features

- **Ensemble Machine Learning Architecture**: Soft-voting ensemble combining **Random Forest**, **Gradient Boosting**, and **XGBoost** with RobustScaler normalization.
- **High Prognostic Discrimination**: ROC AUC of **84.1%** across 11 post-operative clinical complication categories.
- **Interactive Medical UI**: Modern dark-mode clinical dashboard built with Tailwind CSS, SVG circular health score gauge, and responsive design.
- **One-Click Clinical Patient Profiles**:
  - Profile A: Stable / Low Risk
  - Profile B: Moderate Risk / Diabetic & Chronic
  - Profile C: Critical Risk / Prolonged Ischemia & Cirrhosis
- **Clinical Action Guidance**: Real-time evidence-based clinical protocols and key contributing biomarkers for each identified complication vector.
- **Report Export**: Clinical report print and clipboard export capabilities.

---

## 🔬 Multi-Modal Biomarkers (22 Features)

1. **Recipient Demographics & Scores**:
   - `R_Age` (Recipient Age)
   - `R_Gender` (Biological Sex: Male / Female)
   - `R_BMI` (Body Mass Index)
   - `R_MELD_Score` (Model for End-Stage Liver Disease Score)

2. **Pre-Existing Comorbidities & Etiology**:
   - `R_Diabetes` (Pre-transplant Diabetes)
   - `R_Hypertension` (Systemic Hypertension)
   - `R_Alcohol_Abuse` (History of Heavy Alcohol Use)
   - `R_Smoking` (Tobacco History)
   - `R_Hepatitis_B` (HBV Seropositive)
   - `R_Hepatitis_C` (HCV Seropositive)
   - `R_Alcoholic_cirrhosis` (ALD Cirrhosis)
   - `R_Primary_biliary_cirrhosis` (PBC / Autoimmune)

3. **Laboratory Serum Markers & Hematology**:
   - `R_Albumin_level` (Serum Albumin g/L)
   - `R_Na` (Serum Sodium mEq/L)
   - `R_Mg` (Serum Magnesium mg/dL)
   - `R_WBC` (White Blood Cell Count /µL)
   - `R_Lympochyte` (Lymphocyte Percentage %)
   - `R_Platelets` (Platelet Count /µL)

4. **Surgical & Graft Dynamics**:
   - `R_Cold_Ischemia_Time` (Preservation Duration in Minutes)
   - `R_Warm_Ischemia_Time` (Anastomosis Duration in Minutes)
   - `R_Blood_Transfusion` (Intraoperative Transfusion Required)
   - `R_Rejection_Episodes` (Prior Acute Graft Rejection Events)

---

## 🎯 Predicted Complication Classes (11 Target Categories)

- No Complication (Favorable Outcome)
- Primary Graft Non-function
- Artery Thrombosis
- Portal Vein Thrombosis
- Biliary Complications
- Acute Rejection
- Post-transplant Diabetes
- Renal Dysfunction
- Infection
- Metabolic Complications
- Cardiovascular Complications

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Save the Model (Optional - Already Trained)
The ensemble model is already trained and saved in `models/liver_model.pkl`. To retrain:
```bash
python train_model.py
```

### 3. Start the Web Application
```bash
python app.py
```

### 4. Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🛠️ Technology Stack

- **Backend**: Python 3.13, Flask
- **Machine Learning**: Scikit-learn (RandomForest, GradientBoosting, RobustScaler), XGBoost
- **Data Engineering**: Pandas, NumPy
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS, Google Fonts
