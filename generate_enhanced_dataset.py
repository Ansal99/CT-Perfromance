import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load original dataset
df_original = pd.read_csv('data/StudentsPerformance.csv')

# Function to generate realistic enhanced features
def generate_enhanced_features(row):
    # Calculate current overall score
    current_score = (row['math score'] + row['reading score'] + row['writing score']) / 3
    
    # Study hours - correlates with performance
    if current_score >= 85:
        study_hours = np.random.uniform(15, 35)
    elif current_score >= 70:
        study_hours = np.random.uniform(10, 20)
    elif current_score >= 55:
        study_hours = np.random.uniform(5, 15)
    else:
        study_hours = np.random.uniform(2, 10)
    
    # Attendance - generally correlates with performance
    base_attendance = 70 + (current_score / 3)
    attendance = np.clip(base_attendance + np.random.uniform(-10, 10), 50, 100)
    
    # Participation level
    if current_score >= 80:
        participation = np.random.choice(['medium', 'high'], p=[0.2, 0.8])
    elif current_score >= 65:
        participation = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
    elif current_score >= 50:
        participation = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2])
    else:
        participation = np.random.choice(['low', 'medium'], p=[0.7, 0.3])
    
    # Homework completion
    if current_score >= 85:
        homework = np.random.choice(['often', 'always'], p=[0.2, 0.8])
    elif current_score >= 70:
        homework = np.random.choice(['sometimes', 'often', 'always'], p=[0.2, 0.4, 0.4])
    elif current_score >= 55:
        homework = np.random.choice(['rarely', 'sometimes', 'often'], p=[0.2, 0.5, 0.3])
    else:
        homework = np.random.choice(['rarely', 'sometimes'], p=[0.6, 0.4])
    
    # Extra curricular activities
    if current_score >= 75:
        extracurricular = np.random.choice(['yes', 'no'], p=[0.7, 0.3])
    else:
        extracurricular = np.random.choice(['yes', 'no'], p=[0.4, 0.6])
    
    # Sleep hours - healthy sleep correlates with better performance
    if current_score >= 75:
        sleep_hours = np.random.uniform(7, 9)
    elif current_score >= 60:
        sleep_hours = np.random.uniform(6, 8)
    else:
        sleep_hours = np.random.uniform(5, 7)
    
    # Internet access
    internet_access = np.random.choice(['yes', 'no'], p=[0.8, 0.2])
    
    # Previous grade
    if current_score >= 80:
        previous_grade = np.random.choice(['A', 'B'], p=[0.7, 0.3])
    elif current_score >= 70:
        previous_grade = np.random.choice(['A', 'B', 'C'], p=[0.2, 0.6, 0.2])
    elif current_score >= 60:
        previous_grade = np.random.choice(['B', 'C', 'D'], p=[0.2, 0.6, 0.2])
    else:
        previous_grade = np.random.choice(['C', 'D', 'F'], p=[0.3, 0.5, 0.2])
    
    return {
        'current_score': round(current_score, 2),
        'study_hours_per_week': round(study_hours, 1),
        'attendance_rate': round(attendance, 1),
        'participation_level': participation,
        'homework_completion': homework,
        'extracurricular_activities': extracurricular,
        'sleep_hours_per_night': round(sleep_hours, 1),
        'internet_access': internet_access,
        'previous_grade': previous_grade
    }

# Generate enhanced features for all rows
print("Generating enhanced dataset with new features...")
enhanced_features = df_original.apply(generate_enhanced_features, axis=1, result_type='expand')

# Combine with original data
df_enhanced = pd.concat([df_original, enhanced_features], axis=1)

# Add some additional synthetic records to increase dataset size
print("Adding synthetic records to increase dataset size...")

def generate_synthetic_record():
    # Generate random demographics
    gender = random.choice(['male', 'female'])
    ethnicity = random.choice(['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_education = random.choice([
        'some high school', 'high school', 'some college',
        'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'
    ])
    test_prep = random.choice(['completed', 'none'])
    
    # Generate study habits
    study_hours = round(np.random.uniform(2, 35), 1)
    attendance = round(np.random.uniform(60, 100), 1)
    participation = random.choice(['low', 'medium', 'high'])
    homework = random.choice(['rarely', 'sometimes', 'often', 'always'])
    extracurricular = random.choice(['yes', 'no'])
    sleep_hours = round(np.random.uniform(5, 9), 1)
    internet_access = random.choice(['yes', 'no'])
    previous_grade = random.choice(['A', 'B', 'C', 'D', 'F'])
    
    # Calculate expected performance based on factors
    base_score = 50
    
    # Education impact
    education_bonus = {
        'some high school': 0,
        'high school': 5,
        'some college': 10,
        'associate\'s degree': 12,
        'bachelor\'s degree': 15,
        'master\'s degree': 18
    }
    base_score += education_bonus[parental_education]
    
    # Test prep bonus
    if test_prep == 'completed':
        base_score += 8
    
    # Study hours impact
    base_score += (study_hours / 35) * 20
    
    # Attendance impact
    base_score += (attendance - 70) / 3
    
    # Participation impact
    participation_bonus = {'low': -5, 'medium': 0, 'high': 10}
    base_score += participation_bonus[participation]
    
    # Homework impact
    homework_bonus = {'rarely': -8, 'sometimes': -3, 'often': 5, 'always': 10}
    base_score += homework_bonus[homework]
    
    # Previous grade impact
    grade_bonus = {'F': -15, 'D': -8, 'C': 0, 'B': 8, 'A': 15}
    base_score += grade_bonus[previous_grade]
    
    # Add some randomness
    base_score += np.random.uniform(-10, 10)
    base_score = np.clip(base_score, 0, 100)
    
    # Generate individual scores with some correlation
    math_score = int(np.clip(base_score + np.random.uniform(-15, 15), 0, 100))
    reading_score = int(np.clip(base_score + np.random.uniform(-15, 15), 0, 100))
    writing_score = int(np.clip(base_score + np.random.uniform(-15, 15), 0, 100))
    current_score = round((math_score + reading_score + writing_score) / 3, 2)
    
    return {
        'gender': gender,
        'race/ethnicity': ethnicity,
        'parental level of education': parental_education,
        'test preparation course': test_prep,
        'math score': math_score,
        'reading score': reading_score,
        'writing score': writing_score,
        'current_score': current_score,
        'study_hours_per_week': study_hours,
        'attendance_rate': attendance,
        'participation_level': participation,
        'homework_completion': homework,
        'extracurricular_activities': extracurricular,
        'sleep_hours_per_night': sleep_hours,
        'internet_access': internet_access,
        'previous_grade': previous_grade
    }

# Generate 500 additional synthetic records
synthetic_records = [generate_synthetic_record() for _ in range(500)]
df_synthetic = pd.DataFrame(synthetic_records)

# Combine original enhanced data with synthetic data
df_final = pd.concat([df_enhanced, df_synthetic], ignore_index=True)

# Remove 'lunch' column as per requirements
if 'lunch' in df_final.columns:
    df_final = df_final.drop('lunch', axis=1)

# Save the enhanced dataset
df_final.to_csv('data/StudentsPerformance_Enhanced.csv', index=False)

print(f"\n✓ Enhanced dataset created successfully!")
print(f"✓ Total records: {len(df_final)}")
print(f"✓ Original records: {len(df_original)}")
print(f"✓ New synthetic records: {len(df_synthetic)}")
print(f"✓ Total features: {len(df_final.columns)}")
print(f"\nNew features added:")
print("  - current_score (overall average)")
print("  - study_hours_per_week")
print("  - attendance_rate")
print("  - participation_level")
print("  - homework_completion")
print("  - extracurricular_activities")
print("  - sleep_hours_per_night")
print("  - internet_access")
print("  - previous_grade")
print(f"\n✓ Saved to: data/StudentsPerformance_Enhanced.csv")

# Display sample data
print("\nSample records:")
print(df_final.head(3))
print("\nDataset statistics:")
print(df_final[['current_score', 'study_hours_per_week', 'attendance_rate']].describe())