import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

class EnhancedStudentPerformanceModel:
    def __init__(self, data_path='data/StudentsPerformance_Enhanced.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        
    def load_and_generate_data(self):
        """Load data and generate enhanced features if needed"""
        print("="*80)
        print("DATA LOADING AND PREPARATION")
        print("="*80)
        
        # Check if enhanced dataset exists
        if not os.path.exists(self.data_path):
            print("\n⚠️  Enhanced dataset not found. Generating from original data...")
            print("Run: python generate_enhanced_dataset.py first!")
            print("\nAttempting to load original dataset and enhance it...")
            
            try:
                df_original = pd.read_csv('data/StudentsPerformance.csv')
                self.df = self.enhance_original_dataset(df_original)
            except:
                raise FileNotFoundError("Please run 'python generate_enhanced_dataset.py' first!")
        else:
            print("Loading enhanced dataset...")
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded: {len(self.df)} records")
        
        print(f"\nDataset shape: {self.df.shape}")
        print(f"Features: {self.df.columns.tolist()}")
        
        return self.df
    
    def enhance_original_dataset(self, df_original):
        """Enhance original dataset with new features"""
        print("\n📊 Enhancing original dataset with new features...")
        
        # Calculate current overall score
        df_original['current_score'] = (
            df_original['math score'] + 
            df_original['reading score'] + 
            df_original['writing score']
        ) / 3
        
        # Generate study hours
        df_original['study_hours_per_week'] = df_original['current_score'].apply(
            lambda x: np.clip(np.random.uniform(
                2 + (x/10), 5 + (x/5)
            ), 2, 40)
        )
        
        # Generate attendance
        df_original['attendance_rate'] = df_original['current_score'].apply(
            lambda x: np.clip(70 + (x/3) + np.random.uniform(-10, 10), 50, 100)
        )
        
        # Generate participation
        def gen_participation(score):
            if score >= 80: return np.random.choice(['medium', 'high'], p=[0.2, 0.8])
            elif score >= 65: return np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
            else: return np.random.choice(['low', 'medium'], p=[0.6, 0.4])
        
        df_original['participation_level'] = df_original['current_score'].apply(gen_participation)
        
        # Generate homework completion
        def gen_homework(score):
            if score >= 85: return np.random.choice(['often', 'always'], p=[0.2, 0.8])
            elif score >= 70: return np.random.choice(['sometimes', 'often', 'always'], p=[0.2, 0.4, 0.4])
            else: return np.random.choice(['rarely', 'sometimes'], p=[0.5, 0.5])
        
        df_original['homework_completion'] = df_original['current_score'].apply(gen_homework)
        
        # Generate extracurricular activities
        df_original['extracurricular_activities'] = df_original['current_score'].apply(
            lambda x: np.random.choice(['yes', 'no'], p=[0.7, 0.3] if x >= 75 else [0.4, 0.6])
        )
        
        # Generate sleep hours
        df_original['sleep_hours_per_night'] = df_original['current_score'].apply(
            lambda x: np.clip(np.random.uniform(
                7 if x >= 75 else 6,
                9 if x >= 75 else 8
            ), 5, 9)
        )
        
        # Generate internet access
        df_original['internet_access'] = np.random.choice(['yes', 'no'], 
                                                          size=len(df_original), 
                                                          p=[0.85, 0.15])
        
        # Generate previous grade
        def gen_previous_grade(score):
            if score >= 85: return np.random.choice(['A', 'B'], p=[0.8, 0.2])
            elif score >= 75: return np.random.choice(['A', 'B', 'C'], p=[0.2, 0.6, 0.2])
            elif score >= 60: return np.random.choice(['B', 'C', 'D'], p=[0.2, 0.6, 0.2])
            else: return np.random.choice(['C', 'D', 'F'], p=[0.3, 0.5, 0.2])
        
        df_original['previous_grade'] = df_original['current_score'].apply(gen_previous_grade)
        
        # Remove lunch column if exists
        if 'lunch' in df_original.columns:
            df_original = df_original.drop('lunch', axis=1)
        
        print("✓ Enhanced features generated successfully!")
        return df_original
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        print("\nDataset Info:")
        print(f"Total records: {len(self.df)}")
        print(f"Total features: {len(self.df.columns)}")
        
        print("\nNumerical Features Statistics:")
        numerical_cols = ['current_score', 'study_hours_per_week', 'attendance_rate', 
                         'sleep_hours_per_night', 'math score', 'reading score', 'writing score']
        print(self.df[numerical_cols].describe())
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("✓ No missing values!")
        
        print("\nCategorical Features Distribution:")
        categorical_cols = ['participation_level', 'homework_completion', 
                           'extracurricular_activities', 'previous_grade']
        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts())
    
    def preprocess_data(self):
        """Preprocess data and create train/test split"""
        print("\n" + "="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        # All features for prediction
        feature_cols = [
            'gender',
            'race/ethnicity',
            'parental level of education',
            'test preparation course',
            'current score',
            'study hours per week',
            'attendance rate',
            'participation level',
            'homework completion',
            'extracurricular activities',
            'sleep hours per night',
            'internet access',
            'previous grade'
        ]
        
        # Rename columns if needed to match expected format
        column_mapping = {
            'current_score': 'current score',
            'study_hours_per_week': 'study hours per week',
            'attendance_rate': 'attendance rate',
            'participation_level': 'participation level',
            'homework_completion': 'homework completion',
            'extracurricular_activities': 'extracurricular activities',
            'sleep_hours_per_night': 'sleep hours per night',
            'internet_access': 'internet access',
            'previous_grade': 'previous grade'
        }
        
        self.df = self.df.rename(columns=column_mapping)
        
        # Select features and target
        X = self.df[feature_cols]
        y = self.df['writing score']  # Predict writing score
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        print(f"✓ Training set size: {len(self.X_train)}")
        print(f"✓ Test set size: {len(self.X_test)}")
        print(f"\nTarget (writing score) statistics:")
        print(f"  Mean: {self.y_train.mean():.2f}")
        print(f"  Std: {self.y_train.std():.2f}")
        print(f"  Min: {self.y_train.min():.2f}")
        print(f"  Max: {self.y_train.max():.2f}")
        
        # Create preprocessor
        categorical_features = [
            'gender',
            'race/ethnicity',
            'parental level of education',
            'test preparation course',
            'participation level',
            'homework completion',
            'extracurricular activities',
            'internet access',
            'previous grade'
        ]
        
        numerical_features = [
            'current score',
            'study hours per week',
            'attendance rate',
            'sleep hours per night'
        ]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_features)
            ])
        
        # Fit and transform
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"\n✓ Processed feature shape: {self.X_train_processed.shape}")
        print(f"✓ Total features after encoding: {self.X_train_processed.shape[1]}")
    
    def train_models(self):
        """Train multiple models and select the best one"""
        print("\n" + "="*80)
        print("MODEL TRAINING & EVALUATION")
        print("="*80)
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                max_depth=20,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8
            ),
            'Linear Regression': LinearRegression()
        }
        
        best_score = float('inf')
        best_model_name = None
        
        for name, model in models.items():
            print(f"\n{'─'*60}")
            print(f"Training {name}...")
            print(f"{'─'*60}")
            
            model.fit(self.X_train_processed, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_processed)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            print(f"Accuracy (R² as %): {r2*100:.2f}%")
            
            # Cross-validation
            print("\nPerforming 5-fold cross-validation...")
            cv_scores = cross_val_score(model, self.X_train_processed, self.y_train,
                                       cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())
            print(f"Cross-validation RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")
            
            if rmse < best_score:
                best_score = rmse
                best_model_name = name
                self.model = model
        
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   RMSE: {best_score:.4f}")
        print(f"   Accuracy: ~{(1 - (best_score / 100)) * 100:.2f}%")
        print(f"{'='*80}")
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*80)
        print("FINAL MODEL EVALUATION & VISUALIZATION")
        print("="*80)
        
        y_pred = self.model.predict(self.X_test_processed)
        
        # Final metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"\n📊 Final Performance Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Model Accuracy: ~{r2*100:.2f}%")
        
        # Generate visualizations
        print("\n📈 Generating visualizations...")
        self.plot_prediction_scatter(self.y_test, y_pred)
        self.plot_residuals(self.y_test, y_pred)
        
        if hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance()
        
        print("✓ All visualizations saved to outputs/ directory")
    
    def plot_prediction_scatter(self, y_true, y_pred):
        """Plot predicted vs actual scores"""
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5, s=50)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=3, label='Perfect Prediction')
        plt.xlabel('Actual Writing Score', fontsize=13, fontweight='bold')
        plt.ylabel('Predicted Writing Score', fontsize=13, fontweight='bold')
        plt.title('Model Performance: Predicted vs Actual Scores', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(self, y_true, y_pred):
        """Plot residual analysis"""
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residual plot
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5, s=50)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Residual histogram
        ax2.hist(residuals, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importances = self.model.feature_importances_
        
        # Get all feature names
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out([
            'gender', 'race/ethnicity', 'parental level of education',
            'test preparation course', 'participation level', 'homework completion',
            'extracurricular activities', 'internet access', 'previous grade'
        ])
        
        feature_names = [
            'current score', 'study hours per week', 
            'attendance rate', 'sleep hours per night'
        ] + list(cat_features)
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(14, 10))
        plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=13, fontweight='bold')
        plt.ylabel('Features', fontsize=13, fontweight='bold')
        plt.title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self):
        """Save trained model and preprocessor"""
        print("\n" + "="*80)
        print("SAVING MODEL ARTIFACTS")
        print("="*80)
        
        with open('models/student_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('models/preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        print("✓ Model saved: models/student_model.pkl")
        print("✓ Preprocessor saved: models/preprocessor.pkl")
    
    def run_complete_pipeline(self):
        """Execute complete ML pipeline"""
        print("\n" + "="*80)
        print("🎓 STUDENT PERFORMANCE PREDICTION - ENHANCED ML PIPELINE")
        print("="*80)
        print("Training advanced AI model with 12+ comprehensive factors...")
        
        self.load_and_generate_data()
        self.explore_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_model()
        self.save_model()
        
        print("\n" + "="*80)
        print("✓ ✓ ✓ PIPELINE COMPLETED SUCCESSFULLY! ✓ ✓ ✓")
        print("="*80)
        print("\n📁 Generated Files:")
        print("   ✓ models/student_model.pkl (AI Model)")
        print("   ✓ models/preprocessor.pkl (Data Preprocessor)")
        print("   ✓ outputs/prediction_scatter.png (Performance Chart)")
        print("   ✓ outputs/residuals.png (Error Analysis)")
        print("   ✓ outputs/feature_importance.png (Feature Analysis)")
        print("\n🚀 Ready to use! Run: python app.py")
        print("="*80)

if __name__ == '__main__':
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run pipeline
    model = EnhancedStudentPerformanceModel()
    model.run_complete_pipeline()