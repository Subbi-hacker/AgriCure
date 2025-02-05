# AgriCure - Crop Recommendation System

![Crop Recommendation](/static/crop-recommend.jpg)

This project is a culmination of machine learning, frontend, backend, and database integration. The crop recommendation app is made to ease the farmers' dilemma for crop cultivation. Generally, the choice of a crop is decided by the nitrogen, phosphorus, potassium content along with some crucial factors like temperature, humidity, and rainfall (usually measured in mm). Using these inputs, the output of what crop should be grown will be predicted. Along with this, the project also recommends users with the best fertilizer based on the nitrogen, phosphorus, potassium contents, and temperature, moisture, humidity, soil type, and crop type. The user-friendly interface helps users to interact with the website and get recommended with the worthy crop and fertilizer.

**The technologies and libraries used in this project are:**
1. Machine Learning - Decision Tree Classifier
2. Pre Processing - Label Encoder (for string-valued columns in dataset)
3. Python
4. Flask
5. HTML, CSS, and JavaScript - Frontend Website
6. PostgreSQL - Database Integration
7. Confusion Matrix - To plot the comparison between predicted and tested values
8. Libraries: SQLAlchemy, scikit-learn, seaborn, matplotlib, pandas, and flask

**Workflow Diagrams:**
![Technical](/WorkflowDiagram.png)         

![Pest](/PestDetectionFlowchart.png)


The Flask application was deployed on render.com:
1. Build Command: `$pip install -r requirements.txt`
2. Start Command: `$gunicorn -b :$PORT app:app`

**Analysis Report on Crop Recommendation:**
![Report](/static/Report-image.png)

**Here is the link for the website**: [https://crop-recommendation-system-app.onrender.com](https://crop-recommendation-system-app.onrender.com)
