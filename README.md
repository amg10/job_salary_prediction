# Job Salary Prediction
Predict job salary given a posting

# Running the code
Unfortunately, due to the limit on the upload and git push command line feature, the model has not been added to git repository.
Download the models from [here](https://drive.google.com/drive/folders/1dNIrNiXKl3BWYcK_vRYvc2W99h1mxxJh?usp=sharing) and place them into models/ folder.
To run the API, simply run the main.py script.

```
python main.py
```

This will provide a local link to local host with extension 8000. Add '/docs' to access the Swagger UI.
Alternatively, use: https://1afc-142-115-85-198.ngrok-free.app/docs

For the APi, 
1. Expand the API
2. Press the "Try it out" button beside the parameter on right hand side
3. Enter board-name as cohere (Currently configured for cohere)
4. Enter the job id
5. Click execute.

You should see the response in the response body.

# Training the model
To train the model from scratch. 
1. Dowload the training dataset : https://drive.google.com/drive/folders/1dGWN7tHYSEVaWwtHBKf0MHrcal8M-Hoc?usp=sharing
2. Create a data/ folder and place the dataset there
3. Run the following command to start the training

```
python model/train_categorical_random_forest.py
```
There should be a new model and category mapping created in models folder.
