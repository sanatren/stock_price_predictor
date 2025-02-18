github_link :-     https://github.com/sanatren/stock_prices_predictor
(the github link consist of only code files not any models)

google_drive:-  https://drive.google.com/drive/folders/1sNbS_cJihmJmnQgbAiw-k91sRMh00lhC

this following link has full project with trained model and streamlit file to run locally

just open the extracted folder through vscode or any editor

type in terminal: pip install -r requirements.txt 
(to download all dependencies)

then go to streamlit-->app.py

or type in terminal : cd
cd streamlit

run the script app.py in terminal through command :  streamlit run app.py

this will run the project locally

........................................................................................

check out the "data" folder to see how i downloaded and cleaned the yfinance stock data 
go to raw->raw.py (to check the how the data have been downloaded (csv,json) through different data ranges)

go to processed->cleaned.py (cleaned the data )

then go to notebooks-> data_prepration.ipynb folder to check how i preapared the data for model training (ex->such as filling Nan vals ,etc.)

then eda.py.ipynb and pandas_profiling_eda.ipynb for visualization of stocks to find trends ,Volatility
seasonality etc.

also check out pandas_profiling_eda for detailed visualisation

then check the feature_engineering.ipynb where i have extracted and created some important features (such as lagged features to figure out trends ) removed some over dominating feature to avoid overfitting

now the models.ipynb contains models training on ARIMA,SARIMA and many boosting techinques i have trained them on the extracted features engineering data which is insde the raw folder

the ARIMA model was not able to capture seasonality so i used SARIMA which did and performed slightly better .

the boosting models did great while training but not able to capture trends so they were failing at the prediction time as the ML algorithm are not perfect for time-series data

also i trained every model for a seperate stock to perform better.

now, in model evaluation I comapared both ML models , ARIMA and SARIMA
(as you can see the ML models were overfitting)

then in the models folder you can find each variant of model for each stock
both for ARIMA,SARIMA and boosting models.

finally in streamlit-->app.py
i have imported the saved model and used for prediction while also adding the more EDA like heatmap, stock-trends and stationary check , etc.

at predict prices option  you can predict the stock prices for next 7 days.

