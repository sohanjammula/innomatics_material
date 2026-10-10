//Advance Filterings & Clauses//
IN/NOT IN//
IN checks if a value matches any value in a list.
NOT IN excludes values from a list.//


//Customers from the specific countries//

Select CustomerNumber, CustomerName, Country from customers 
where Country IN ('USA','France','Australia');

//NOT IN//

Select CustomerNumber, CustomerName, Country from customers 
where Country NOT IN ('USA','France');

//Sales team wants to focus Only cars and motorcycles exciluding Ships and trains //

Select * from Products;

Select productCode, ProductName, ProductLine from Products 
where ProductLine NOT IN('ships','Trains');

//Between - Filters values within a range (inclusive of boundaries).//
//Procusts Priced between 50 and 100//

Select productCode, ProductName,buyPrice from Products
where buyPrice Between 50 AND 100;

//LIke - Used for pattern matching with wildcards//

//Identify all customers t=with brand Name starting with "MINI//

Select CustomerNumber, customerName from Customers 
where customerName LIKE 'Mini%';

Select * from Products;

//Find the Products containing 'Ford'//
Select ProductCode, ProductName from Products
where ProductName LIKE '%Ford%';
//Producr Name End with Truck//

Select ProductCode, ProductName from Products
where ProductName LIKE '%Truck';

//Distinct - Returns only unique values, removing duplicates.//

//List all Unique countries of customers//

Select DISTINCT country
from Customers;

Select * from Products;
//Find the Unique Product Line//

Select Distinct ProductLine
from Products;

Select * from Products;

//LIMIT - Restricts the number of rows returned.//

// Top 5 Most expensive Products//

Select ProductCode, ProductName, MSRP
from Products
Order By MSRP DESC
LIMIT 10;
Select *from Products;
//Shows Products 6 to 10 buyPrice//

//OFFSET- Skips a specified number of rows before starting to return results. Often used with LIMIT for pagination.//

Select ProductCode, productName, MSRP
from Products
Order by MSRP DESC
LIMIT 5 
OFFSET 5;


//Find the customers from USA and FRance whose credit Limit is Between 50000 and 1,50,000//
//customerCode, CustomeName, Country, creditLimit//
Select customerNumber, customerName, country, creditLimit from customers
where country IN ('USA','France') and
CreditLimit Between 50000 and 150000;

//Management wants to review products fro Pricing stategy//
//condition - ProductLine muct be Classic Cars and Motorcycles
//MSRP must be between 100 to 200
//Product Name should include 'Ford'//

Select productCode, productName, productLine, MSRP from Products 
where ProductLine IN('Classic Cars','Motorcycles')
AND MSRP Between 100 AND 200
AND ProductName LIKE '%Ford%';
//Clauses --
//Group By
Having,
Order by//
//Group By - GROUP BY is used to arrange identical data into groups. It’s often paired with aggregate functions to summarize data//
//Finance team wantsto understand How munch each customer has Paid in total //

Select * from Payments;

Select CustomerNumber, SUM(amount) as totalPaymnet from Payments
GROUP By CustomerNumber;

Select * from Orders;

// Management wants to see How many orderd are shipped , cancelled and on hold//

Select Status,Count(orderNumber) as total_Orders
from Orders
Group by Status;

//Aggregated functions : Aggregate functions perform calculations on a set of values and return a single value. Common ones: SUM, AVG, COUNT, MIN, MAX//

Select SUM(buyPrice) as totalprice
from Products ;

SELECT country, COUNT(customerNumber) AS totalCustomers
FROM Customers

//HAVING- HAVING is used to filter groups created by GROUP BY. Unlike WHERE, which filters rows, HAVING filters aggregated results//

