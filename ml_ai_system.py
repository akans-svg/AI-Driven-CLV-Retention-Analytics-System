
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, r2_score

def train_models():
    df = pd.read_csv("customers_large.csv")
    
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    X = df[['Age','Tenure','MonthlySpend','SubscriptionType']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    churn_model = RandomForestClassifier()
    churn_model.fit(X_train, y_train)
    print(classification_report(y_test, churn_model.predict(X_test)))

    joblib.dump(churn_model, "saved_models/churn_model.pkl")

    clv_model = RandomForestRegressor()
    clv_model.fit(X, df['CLV'])
    print("CLV R2:", r2_score(df['CLV'], clv_model.predict(X)))

    joblib.dump(clv_model, "saved_models/clv_model.pkl")

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Segment'] = kmeans.fit_predict(X)
    df.to_csv("customers_segmented.csv", index=False)

if __name__ == "__main__":
    train_models()
