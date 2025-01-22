from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import joblib
import uvicorn
import pprint
from fastapi import FastAPI, HTTPException

app = FastAPI()
driver = webdriver.Chrome()
model = joblib.load('model/random_forest_cat_salary_model.pkl')
category_mappings = joblib.load('model/category_mappings.joblib')

# Helper function to get the job posting details
def fetch_job_posting(board_name: str, postingid: str):
    if board_name=="cohere":
        url = f"https://jobs.ashbyhq.com/cohere/{postingid}"

        #try:
        driver.get(url)
        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")

        job_title = soup.find("h1", {"class": "_title_ud4nd_34"}).text.strip()
        description = soup.find("div",{"class":"_descriptionText_4fqrp_201"}).text.strip()
        location = soup.find('h2', text='Location').find_next('p').text.strip()
        contract_type = soup.find('h2', text='Employment Type').find_next('p').text.strip().replace(" ","_").lower()
        department = soup.find('h2', text='Department').find_next('p').text.strip()
        company_name = "Cohere"
        
        return {
            "job_title": job_title,
            "description": description,
            "location": location,
            "contract_type": contract_type,
            "company": company_name,
            "department": department,
        }

        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=f"Error fetching job posting: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"Provided board name not supported.")
    

# Preprocess job details for the model
def preprocess_job_details(job_details):
    # Convert job details to a DataFrame row (ensure it matches model training schema)
    df = pd.DataFrame([{
        "Title": job_details["job_title"],
        "FullDescription": job_details["description"],
        "LocationRaw": job_details["location"],
        "ContractType": job_details["contract_type"],
    }])
    
    for label in df.columns:
        if label in category_mappings:
            # Apply the exact category mapping from the training data
            df[label] = pd.Categorical(df[label], categories=category_mappings[label]).codes + 1
    
    return df

# Define FastAPI endpoint
@app.get("/predict/salary/{board_name}/{postingid}")
async def predict_salary(board_name: str, postingid: str):
    try:
        # Fetch job details
        job_details = fetch_job_posting(board_name, postingid)
        if not job_details:
            raise HTTPException(status_code=404, detail="Job posting not found")

        # Preprocess job details

        input_features = preprocess_job_details(job_details)

        # Predict salary
        predicted_salary = model.predict(input_features)[0]
        
        # Return response
        return f"The predicted salary is: ${predicted_salary:.2f}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Define the main function to run Uvicorn
def main():
    # Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Check if the script is being executed directly, then call the main function
if __name__ == "__main__":
    main()