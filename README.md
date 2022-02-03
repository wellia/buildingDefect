# Building Defect Repository

Task 1: Feasibility study to explore getting building defect data online using Web Scraping techniques. 
Implemented Python libraries such as Http requests, BeautifulSoups and Selenium. (https://github.com/wellia/WebScrapping)

Task 2: Cleaned, analysed and mined the data with Python, Spacy, NLTK (https://github.com/wellia/buildingDefect/blob/main/buildingDefect_process.py)
Visualized the data with tableau BuildingDefect_general, BuildingDefect_responseDays, BuildingDefect_elements dashboards (https://public.tableau.com/app/profile/wellia.lioeng#!/)

Task 3: build models to predict the building defect categories. 
- Performed data analysis.
- Cleaned the data, selected categories and performed data augmentation
- Trained the model with ktrain (https://github.com/amaiya/ktrain#overview), a Lightweight Neural Network wrapper for Keras. The model is trained with DistilBERT, light and faster transformer model based on BERT. (https://paperswithcode.com/method/distillbert). The accuracy is 73%. (https://github.com/wellia/buildingDefect/blob/main/BuildingDefect_model_ktrain.ipynb)
- Predicted categories for new data
- Experimented with machine learning xgboost, random forest (https://github.com/wellia/buildingDefect/blob/main/buildingDefect_model_ML.py)
