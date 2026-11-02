import os
import pickle
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Clinical recommendations and severity mapping for each post-transplant complication
CLINICAL_RECOMMENDATIONS = {
    'Artery Thrombosis': {
        'severity': 'Critical',
        'recommendation': 'Urgent hepatic artery Doppler ultrasonography. Assess resistive index (<0.5 or >0.8 requires exploration). Monitor AST/ALT spikes and consider urgent revascularization or anticoagulation protocol.',
        'badge': 'rose'
    },
    'Biliary Complications': {
        'severity': 'High Risk',
        'recommendation': 'Obtain MRCP/ERCP to evaluate for anastomotic stricture, biliary leakage, or bile duct necrosis. Check total bilirubin and alkaline phosphatase serial trends.',
        'badge': 'amber'
    },
    'Cardiovascular Complications': {
        'severity': 'High Risk',
        'recommendation': 'Continuous cardiac telemetry monitoring. Schedule urgent 12-lead ECG and transthoracic echocardiogram. Optimize arterial pressure and volume homeostasis.',
        'badge': 'red'
    },
    'Infection': {
        'severity': 'Moderate-High',
        'recommendation': 'Send full infectious workup: blood, urine, bile/sputum cultures. Check procalcitonin and CRP. Institute targeted antimicrobial and antifungal coverage.',
        'badge': 'orange'
    },
    'Metabolic Complications': {
        'severity': 'Moderate',
        'recommendation': 'Assess lipid profile, electrolyte panels (potassium, phosphate, magnesium), and acid-base balance. Implement nutritional support and modify immunosuppression if hypermetabolic.',
        'badge': 'yellow'
    },
    'No Complication': {
        'severity': 'Favorable',
        'recommendation': 'Excellent post-operative prognostic trajectory. Maintain maintenance immunosuppressive regimen (tacrolimus/mycophenolate), routine graft function surveillance, and regular outpatient follow-up.',
        'badge': 'emerald'
    },
    'Portal Vein Thrombosis': {
        'severity': 'Critical',
        'recommendation': 'Urgent portomesenteric duplex ultrasound or dynamic CT venography. Monitor for sudden ascites or acute graft dysfunction. Assess need for catheter-directed thrombolysis or thrombectomy.',
        'badge': 'purple'
    },
    'Post-transplant Diabetes': {
        'severity': 'Moderate',
        'recommendation': 'Intensive capillary blood glucose surveillance (QID). Initiate subcutaneous basal-bolus insulin protocol. Coordinate with endocrinology and evaluate tapering systemic corticosteroids.',
        'badge': 'blue'
    },
    'Primary Graft Non-function': {
        'severity': 'Emergency',
        'recommendation': 'EMERGENCY: Immediate alert to transplant surgical team and relisting for urgent re-transplantation (Status 1A). Monitor severe coagulopathy (INR >2.5), profound acidosis, and refractory hypoglycemia.',
        'badge': 'red'
    },
    'Rejection': {
        'severity': 'High Risk',
        'recommendation': 'Urgent percutaneous liver biopsy to assess Banff schema acute cellular rejection score (RAI). Check trough tacrolimus/cyclosporine levels. Prepare pulsed methylprednisolone therapy.',
        'badge': 'violet'
    },
    'Renal Dysfunction': {
        'severity': 'High Risk',
        'recommendation': 'Serial serum creatinine, BUN, and urine output tracking. Adjust calcineurin inhibitor dosage (or consider mTOR-inhibitor switch). Avoid NSAIDs and nephrotoxic IV contrast.',
        'badge': 'cyan'
    }
}

# Load the model and metadata
model_path = os.path.join(os.path.dirname(__file__), 'models', 'liver_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print("Model loaded successfully!")
    print(f"Model accuracy: {model_data.get('accuracy', 0):.3f}")
    print(f"Number of features: {len(model_data.get('features', []))}")
    print(f"Classes: {model_data.get('classes', [])}")
except Exception as e:
    print(f"Error loading model from {model_path}: {str(e)}")
    model_data = None

def compute_assessment(probabilities, classes, feature_importance):
    """Compute rich clinical prognostic details based on model probability distribution"""
    top_idx = int(np.argmax(probabilities))
    top_class = classes[top_idx]
    top_prob = float(probabilities[top_idx])

    # Find probability of 'No Complication'
    no_comp_prob = 0.0
    for idx, c in enumerate(classes):
        if c.strip().lower() == 'no complication':
            no_comp_prob = float(probabilities[idx])
            break

    adverse_prob = 1.0 - no_comp_prob

    # Medical classification based on predicted outcome and probability
    critical_complications = {'Primary Graft Non-function', 'Artery Thrombosis', 'Portal Vein Thrombosis', 'Cardiovascular Complications', 'Rejection'}
    moderate_complications = {'Post-transplant Diabetes', 'Metabolic Complications', 'Infection', 'Biliary Complications', 'Renal Dysfunction'}

    if top_class.strip().lower() == 'no complication':
        risk_level = "Low Risk"
        health_score = min(98.5, max(80.0, 75.0 + (top_prob * 25.0)))
    elif top_class in critical_complications:
        risk_level = "High Risk"
        health_score = max(10.0, min(50.0, 60.0 - (top_prob * 65.0)))
    elif top_class in moderate_complications:
        if top_prob >= 0.45:
            risk_level = "High Risk"
            health_score = max(25.0, min(60.0, 70.0 - (top_prob * 55.0)))
        else:
            risk_level = "Moderate Risk"
            health_score = max(45.0, min(75.0, 85.0 - (top_prob * 50.0)))
    else:
        risk_level = "Moderate Risk"
        health_score = 65.0

    # Build detailed complications breakdown
    complications = []
    for i, prob in enumerate(probabilities):
        prob_val = float(prob)
        c_name = classes[i]
        rec = CLINICAL_RECOMMENDATIONS.get(c_name, {
            'severity': 'Moderate',
            'recommendation': 'Standard clinical monitoring and follow-up recommended.',
            'badge': 'slate'
        })

        if prob_val >= 0.05 or i == top_idx:
            # Top contributing factors
            top_factors = [f['feature'] for f in feature_importance[:4]] if feature_importance else []
            complications.append({
                'name': c_name,
                'probability': round(prob_val, 4),
                'probability_percent': round(prob_val * 100, 1),
                'description': f"Risk of {c_name.lower()} based on patient clinical biomarkers",
                'severity': rec['severity'],
                'recommendation': rec['recommendation'],
                'badge': rec['badge'],
                'contributing_factors': top_factors
            })

    complications.sort(key=lambda x: x['probability'], reverse=True)

    # Full distribution for all classes
    all_predictions = [
        {
            'name': classes[i],
            'probability': round(float(probabilities[i]), 4),
            'probability_percent': round(float(probabilities[i]) * 100, 1)
        }
        for i in range(len(classes))
    ]
    all_predictions.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'top_prediction': top_class,
        'top_probability': round(top_prob * 100, 1),
        'risk_level': risk_level,
        'health_score': round(health_score, 1),
        'complications': complications,
        'all_predictions': all_predictions
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None
    })

@app.route('/api/model-info')
def model_info():
    if not model_data:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    return jsonify({
        'status': 'success',
        'accuracy': round(model_data.get('accuracy', 0.452) * 100, 1),
        'roc_auc': round(model_data.get('roc_auc', 0.841) * 100, 1),
        'features_count': len(model_data.get('features', [])),
        'features': model_data.get('features', []),
        'classes': model_data.get('classes', []),
        'top_features': model_data.get('feature_importance', [])[:8]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_data:
            return jsonify({
                'status': 'error',
                'message': 'Model is not loaded. Please train the model first.'
            }), 500

        # Get input data from JSON or form
        form_data = request.get_json(silent=True) or request.form.to_dict()
        if not form_data:
            return jsonify({
                'status': 'error',
                'message': 'No patient data received in request.'
            }), 400

        # Field mapping
        field_mapping = {
            'r_age': 'R_Age',
            'r_bmi': 'R_BMI',
            'r_meld_score': 'R_MELD_Score',
            'r_diabetes': 'R_Diabetes',
            'r_hypertension': 'R_Hypertension',
            'r_alcohol_abuse': 'R_Alcohol_Abuse',
            'r_smoking': 'R_Smoking',
            'r_hepatitis_b': 'R_Hepatitis_B',
            'r_hepatitis_c': 'R_Hepatitis_C',
            'r_albumin_level': 'R_Albumin_level',
            'r_na': 'R_Na',
            'r_mg': 'R_Mg',
            'r_wbc': 'R_WBC',
            'r_lympochyte': 'R_Lympochyte',
            'r_platelets': 'R_Platelets',
            'r_cold_ischemia_time': 'R_Cold_Ischemia_Time',
            'r_warm_ischemia_time': 'R_Warm_Ischemia_Time',
            'r_blood_transfusion': 'R_Blood_Transfusion',
            'r_alcoholic_cirrhosis': 'R_Alcoholic_cirrhosis',
            'r_primary_biliary_cirrhosis': 'R_Primary_biliary_cirrhosis',
            'r_gender': 'R_Gender',
            'r_rejection_episodes': 'R_Rejection_Episodes'
        }

        defaults = {
            'R_Age': 50,
            'R_BMI': 25.0,
            'R_MELD_Score': 15,
            'R_Albumin_level': 40.0,
            'R_Na': 140.0,
            'R_Mg': 2.0,
            'R_WBC': 7000.0,
            'R_Lympochyte': 30.0,
            'R_Platelets': 200000.0,
            'R_Cold_Ischemia_Time': 360.0,
            'R_Warm_Ischemia_Time': 30.0,
            'R_Rejection_Episodes': 0,
            'R_Gender': 0,
            'R_Diabetes': 0,
            'R_Hypertension': 0,
            'R_Alcohol_Abuse': 0,
            'R_Smoking': 0,
            'R_Hepatitis_B': 0,
            'R_Hepatitis_C': 0,
            'R_Alcoholic_cirrhosis': 0,
            'R_Primary_biliary_cirrhosis': 0,
            'R_Blood_Transfusion': 0
        }

        input_data = {}
        for form_field, model_field in field_mapping.items():
            value = form_data.get(form_field)
            if value is None or value == '':
                value = defaults[model_field]
            elif model_field == 'R_Gender':
                val_str = str(value).strip().lower()
                value = 1 if val_str in ['1', 'male', 'm', 'true'] else 0
            elif model_field in ['R_Diabetes', 'R_Hypertension', 'R_Alcohol_Abuse', 'R_Smoking',
                                 'R_Hepatitis_B', 'R_Hepatitis_C', 'R_Alcoholic_cirrhosis',
                                 'R_Primary_biliary_cirrhosis', 'R_Blood_Transfusion', 'R_Rejection_Episodes']:
                try:
                    val_str = str(value).strip().lower()
                    if val_str in ['true', 'yes', 'on']:
                        value = 1
                    elif val_str in ['false', 'no', 'off']:
                        value = 0
                    else:
                        value = int(float(value))
                except (ValueError, TypeError):
                    value = defaults[model_field]
            else:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = defaults[model_field]
            input_data[model_field] = value

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all model features exist
        for feature in model_data['features']:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[model_data['features']]

        # Scale features
        input_df_scaled = pd.DataFrame(
            model_data['scaler'].transform(input_df),
            columns=input_df.columns
        )

        # Predict
        probabilities = model_data['model'].predict_proba(input_df_scaled)[0]
        assessment = compute_assessment(
            probabilities,
            model_data['classes'],
            model_data.get('feature_importance', [])
        )

        return jsonify({
            'status': 'success',
            'risk_level': assessment['risk_level'],
            'health_score': assessment['health_score'],
            'predicted_complication': assessment['top_prediction'],
            'top_probability': assessment['top_probability'],
            'complications': assessment['complications'],
            'all_predictions': assessment['all_predictions']
        })

    except Exception as e:
        import traceback
        print("Error during prediction:", traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Prediction computation error: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Liver Transplant Disease Prediction App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
