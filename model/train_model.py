import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error


df_train = pd.read_csv('data/Train_rev1.zip', compression='zip', header=0, sep=',', quotechar='"')


# Handle empty target values
df_train['SalaryNormalized'] = df_train['SalaryNormalized'].fillna(df_train['SalaryNormalized'].median())
df_train['Title'] = df_train['Title'].fillna('')
df_train['FullDescription'] = df_train['FullDescription'].fillna('')
df_train['LocationNormalized'] = df_train['LocationNormalized'].fillna('')
df_train['ContractType'] = df_train['ContractType'].fillna('')

# Features available from site are limited. 
# Thus corresponding features from dataset
features = ['Title', 'FullDescription', 'LocationNormalized', 'ContractType']
target = 'SalaryNormalized'

df_train['ContractType'] = df_train['ContractType'].astype('category').cat.as_ordered()
df_train['LocationNormalized'] = df_train['LocationNormalized'].astype('category').cat.as_ordered()

vectorizer = TfidfVectorizer(max_features=500)
df_train['FullDescription'] = vectorizer.fit_transform(df_train[['FullDescription']])
df_train['Title'] = vectorizer.fit_transform(df_train[['Title']])

for label,content in df_train.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_train[label] = pd.Categorical(content).codes+1

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Split data into train and test sets
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_train[features], df_train[target], test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Save the pipeline for future use (optional)
joblib.dump(model, 'random_forest_salary_model.pkl')
