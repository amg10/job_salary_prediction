import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df_train = pd.read_csv('data/Train_rev1.zip', compression='zip', header=0, sep=',', quotechar='"')
category_mappings = {}

for label, content in df_train.items():
    if pd.api.types.is_object_dtype(content):
        df_train[label] = content.astype("category").cat.as_ordered()
        category_mappings[label] = df_train[label].cat.categories

for label,content in df_train.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_train[label] = pd.Categorical(content).codes+1

df_tmp = df_train.copy()

#Features available from site are limited. 
#Thus corresponding features from dataset
features = ['Title', 'FullDescription', 'LocationRaw', 'ContractType']
target = 'SalaryNormalized'

X = df_train[features]
y = df_train[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor(n_jobs=-1)
model.fit(X_train,y_train)


# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Save the pipeline for future use (optional)
joblib.dump(model, 'model/random_forest_cat_salary_model.pkl')
joblib.dump(category_mappings, 'model/category_mappings.joblib')