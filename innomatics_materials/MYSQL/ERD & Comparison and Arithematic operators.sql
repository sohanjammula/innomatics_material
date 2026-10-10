
Use classicmodels;

Select * from Customers;
Select *from Employees;
Select * from products;

--//Relationship Types in MySQL:
One-to-One (1:1) : Each row in Table A relates to exactly one row in Table B.
One-to-Many (1:N) : One row in Table A relates to multiple rows in Table B.
Many-to-Many (M:N) : Rows in Table A relate to multiple rows in Table B, and vice versa.
Self-Referencing: A table relates to itself.
//--
//Comparison Operators :
Comparison operators are used in the WHERE clause to compare column values against constants or other columns. They return TRUE/FALSE and filter rows accordingly.
Operators include:
•	= (equal to)
•	!= or <> (not equal to)
•	< (less than)
•	> (greater than)
•	<= (less than or equal to)
•	>= (greater than or equal to)
//--
--Scenarios fro Comparision Operators :--

//Wharehouse team wants to identify Products running low in stock //

Select Productcode, ProductName, quantityInStock
from Products
where quantityInStock <100;

//Customers with credit limit greater then 1,00,000//
//Sales team wants to target high values customers - those who have cresit limit greater then 1,00,000//

Select customerNumber, CustomerName, Country, CreditLimit
from Customers
where creditLimit >100000;

//Logistcs team wants to track delayed shipments //

Select orderNumber, OrderDate, requiredDate, ShippedDate from Orders
where shippedDate >requiredDate;
//Customers not from USA//
Select CustomerNumber, CustomerName, country
from Customers
where country <>'USA';

//Finance team wants to analyze Mid range Price Prodcuts - Products with MSRP between 100 to 200//
Select ProductCode, ProductName, MSRP from Products
Where MSRP >=100 AND MSRP <= 200;

--//Arithmetic Operators –
Arithmetic operators allow you to perform mathematical calculations inside SQL queries. Operators:
•	+ (addition)
•	- (subtraction)
•	* (multiplication)
•	/ (division)//--
Select * from Products;

//Calcualte Profit Margin for Products //

Select ProductCode, ProductName, MSRP-buyPrice as ProfitMargin
from Products;

Select * from orderdetails;

Select OrderNumber, ProductCode, quantityOrdered*priceEach as Sales
from orderdetails;

