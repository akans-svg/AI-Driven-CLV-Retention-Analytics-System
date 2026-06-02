
CREATE TABLE customers (
    CustomerID INT PRIMARY KEY,
    Age INT,
    Tenure INT,
    MonthlySpend FLOAT,
    SubscriptionType INT,
    TotalSpend FLOAT,
    Churn INT,
    CLV FLOAT
);

-- KPI Queries
SELECT COUNT(*) AS Total_Customers FROM customers;
SELECT AVG(CLV) AS Average_CLV FROM customers;
SELECT SUM(CASE WHEN Churn=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS Churn_Rate FROM customers;
