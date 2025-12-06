from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

def load_model():
    try:
        with open('models/student_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print("✓ Model loaded successfully")
        return model, preprocessor
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return None, None

model, preprocessor = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not model or not preprocessor:
            return jsonify({'success': False, 'error': 'Model not loaded. Run train_model.py first.'})
        
        # Prepare input data with new features
        input_data = pd.DataFrame([{
            'gender': data['gender'],
            'race/ethnicity': data['ethnicity'],
            'parental level of education': data['parental_education'],
            'test preparation course': data['test_prep'],
            'current score': float(data['current_score']),
            'study hours per week': float(data['study_hours']),
            'attendance rate': float(data['attendance']),
            'participation level': data['participation'],
            'homework completion': data['homework']
        }])
        
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        predicted_score = float(prediction)
        
        current_score = float(data['current_score'])
        
        # Calculate improvement potential
        improvement = predicted_score - current_score
        
        def get_category(score):
            if score >= 80: return 'Excellent'
            elif score >= 70: return 'Good'
            elif score >= 60: return 'Average'
            elif score >= 50: return 'Below Average'
            else: return 'Needs Improvement'
        
        # Generate personalized recommendations
        recommendations = []
        
        if float(data['study_hours']) < 10:
            recommendations.append('Increase study hours to at least 10 hours per week')
        
        if float(data['attendance']) < 90:
            recommendations.append('Improve attendance - aim for 95%+ attendance rate')
        
        if data['participation'] in ['low', 'medium']:
            recommendations.append('Increase class participation and engagement')
        
        if data['homework'] in ['rarely', 'sometimes']:
            recommendations.append('Complete homework assignments regularly')
        
        if data['test_prep'] == 'none':
            recommendations.append('Consider enrolling in test preparation courses')
        
        if not recommendations:
            recommendations = ['Maintain excellent performance', 'Consider advanced coursework', 'Help peers through tutoring']
        
        return jsonify({
            'success': True,
            'predicted_score': round(predicted_score, 2),
            'current_score': round(current_score, 2),
            'improvement_potential': round(improvement, 2),
            'performance_category': get_category(predicted_score),
            'recommendations': recommendations,
            'confidence': round(min(100, max(70, 100 - abs(improvement) * 3)), 2)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/student_performance', methods=['POST'])
def student_performance():
    try:
        data = request.get_json()
        
        student_info = {
            'name': data.get('name', 'Student'),
            'gender': data['gender'],
            'ethnicity': data['ethnicity'],
            'parental_education': data['parental_education'],
            'test_prep': data['test_prep'],
            'study_hours': float(data['study_hours']),
            'attendance': float(data['attendance']),
            'participation': data['participation'],
            'homework': data['homework']
        }
        
        current_score = float(data['current_score'])
        
        # Generate realistic historical data based on current performance
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        history = []
        
        # Create growth pattern based on study habits
        base_growth = 0.7
        if student_info['study_hours'] > 15:
            base_growth = 0.6
        elif student_info['study_hours'] < 5:
            base_growth = 0.8
        
        for i, month in enumerate(months):
            progress = (i + 1) / len(months)
            trend_score = current_score * (base_growth + 0.3 * progress)
            
            # Add some realistic variation
            variation = np.random.randint(-3, 4)
            score = max(0, min(100, int(trend_score + variation)))
            
            history.append({
                'month': month,
                'score': score
            })
        
        # Calculate statistics
        def get_category(score):
            if score >= 80: return 'Excellent'
            elif score >= 70: return 'Good'
            elif score >= 60: return 'Average'
            elif score >= 50: return 'Below Average'
            else: return 'Needs Improvement'
        
        # Identify strengths
        strengths = []
        if student_info['study_hours'] >= 15:
            strengths.append('Dedicated study routine')
        if student_info['attendance'] >= 95:
            strengths.append('Excellent attendance')
        if student_info['participation'] == 'high':
            strengths.append('Active class participation')
        if student_info['homework'] == 'always':
            strengths.append('Consistent homework completion')
        if student_info['test_prep'] == 'completed':
            strengths.append('Completed test preparation')
        
        if not strengths:
            strengths = ['Room for improvement in all areas']
        
        # Generate recommendations
        recommendations = []
        if student_info['study_hours'] < 10:
            recommendations.append('Increase weekly study hours to 10-15 hours')
        if student_info['attendance'] < 90:
            recommendations.append('Improve attendance - target 95%+ rate')
        if student_info['participation'] == 'low':
            recommendations.append('Increase active participation in class')
        if student_info['homework'] in ['rarely', 'sometimes']:
            recommendations.append('Complete all homework assignments')
        if student_info['test_prep'] == 'none':
            recommendations.append('Enroll in test preparation program')
        if current_score < 70:
            recommendations.append('Consider additional tutoring support')
        
        if not recommendations:
            recommendations = ['Maintain excellent habits', 'Consider mentoring other students', 'Explore advanced coursework']
        
        # Predict future performance
        if not model or not preprocessor:
            predicted_future = current_score + 5
        else:
            try:
                predict_data = pd.DataFrame([{
                    'gender': student_info['gender'],
                    'race/ethnicity': student_info['ethnicity'],
                    'parental level of education': student_info['parental_education'],
                    'test preparation course': student_info['test_prep'],
                    'current score': current_score,
                    'study hours per week': student_info['study_hours'],
                    'attendance rate': student_info['attendance'],
                    'participation level': student_info['participation'],
                    'homework completion': student_info['homework']
                }])
                predict_processed = preprocessor.transform(predict_data)
                predicted_future = float(model.predict(predict_processed)[0])
            except:
                predicted_future = current_score + 5
        
        statistics = {
            'current_score': int(current_score),
            'predicted_score': round(predicted_future, 2),
            'performance_category': get_category(current_score),
            'improvement_potential': round(predicted_future - current_score, 2),
            'strengths': strengths,
            'recommendations': recommendations,
            'study_efficiency': 'High' if student_info['study_hours'] >= 15 else 'Medium' if student_info['study_hours'] >= 10 else 'Low',
            'engagement_level': student_info['participation'].capitalize()
        }
        
        return jsonify({
            'success': True,
            'student_info': student_info,
            'current_score': int(current_score),
            'history': history,
            'statistics': statistics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    if not os.path.exists('models/student_model.pkl'):
        print("\n⚠️  Run 'python train_model.py' first!\n")
    app.run(debug=True, port=5000)