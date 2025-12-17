# Project Description 
Analysis and Predictive Modeling of Crime in Chicago (2001–Present)
- Bigdata Final Project for NYU Tandon Bigdata

# Team Member
Yuchen Liu\
Haoyang Lin\
Han Wen\
Leile Zhang

# Instructions
1. Download all the project files
2. Run `docker-compose build` in folder
3. Start by using `docker-compose up -d`
4. Open jupyter at `localhost:8888`

# Data Ingestion
Open folder `/spark-apps` \
Run each file in the order:
1. `kafka_producer.py`
2. `stream_kafka_to_hdfs.py`

Note: kafka_producer.py **DEFAULT** get all data from API, not CSV \
IF you want to use CSV file to speed up data ingestion, place the CVS file inside of `/data` folder, then run `kafka_producer.py --mode csv` \
IF you want to get the newest data, use `kafka_producer.py --mode incremental` \
IF you want to get the newest data automatically, use `kafka_producer.py --mode continuous`

# Data Transformation
Open folder `/spark-apps`\
Run `chicago_crimes_etl.py`

# ML
Open folder `/notebooks`\
Run `ml_crimeTypes_predict.py`, `ml_domestic_predict.py`, `ml_isArrested_predict.py`

# Analysis and Visualization
Open folder `/notebooks`\
Run `Chicago Crime Analytics & Visualization.ipynb` to see the output\
Run `basic_heatmap.ipynb` and `heatmap_community.py` to see heatmap\
The output graphs are in folder `/notebooks/outputs`\

Run `ml_vizs.ipynb` to see the visualization of ML output and prediction

# Stop
Run `docker-compose down`