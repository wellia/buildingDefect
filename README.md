# Building Defect Repository

The first task is to do feasibility study to explore getting building defect data online using Web Scraping techniques. 
I used Python libraries such as Http requests, BeautifulSoups and Selenium. Code: (https://github.com/wellia/WebScrapping)

The second task is to build a model to predict the building defect categories. 
- I performed data analysis. The result was inconsistent data entry and highly imbalanced data
- I cleaned the data, selected categories and performed data augmentation
- To train a model, I implemented ktrain (https://github.com/amaiya/ktrain#overview), a Lightweight Neural Network wrapper for Keras. The model is trained with DistilBERT, light and faster transformer model based on BERT. (https://paperswithcode.com/method/distillbert). The accuracy is 90%
- Using this model, I predicted categories for new data
