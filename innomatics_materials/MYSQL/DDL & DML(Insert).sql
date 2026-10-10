
Create Database Innomatics;
Use Classicmodels;

Select * from customers;

Use innomatics;
//--Create table syntax
CREATE TABLE TABLE_NAME (
COLUMN_NAME1 WITH DATATYPE,
COLUMN_NAME2 WITH DATATYPE,
COLUMN_NAME3 WITH DATATYPE
)--//

CREATE TABLE Employees(
EmpID INT,
EmpName Varchar (50),
Salary Decimal(10,2),
DOJ Date
);

Select * from Department ;

CREATE TABLE Department(
DeptID INT,
DeptName Varchar (50),
Location varchar(50)
);

//-- ALTER
Used to Modify existing table structure--
Syntax: 
ALTER TABLE table_name
ADD COLUMN Coulumn_Name dataType;//

ALTER TABLE Employees
ADD COLUMN Email varchar(50);

//-- Syntax for coulumn modification//
//ALTER TABLE Table_name
Modify column column_name new_datatype;--//

ALTER TABLE Employees
MODIFY COLUMN Salary INT;
//Syntax for change the column Name--//
ALTER TABLE Table_name
CHANGE Old_column_Name New_column_Name datatype; --//

ALTER TABLE Employees
CHANGE EmpName EMP_FirstName varchar (50);

ALTER TABLE Employees
ADD COLUMN Emp_LastName varchar(50);


//DROP -- Permanently removes the entire table from the database or Entire database fromthe server --//
Syntax - DROP DATABASE Database_Name
DROP TABLE Table_Name--//

use first_database;
Drop table department;
Drop database first_database;

Use Innomatics;

Select * from Employees;


--//Insert - Used to insert the data into table --//
Syntax - INSERT INTO Table_Name( COLUMN1, COLUMN2, COLUMN3)
Values ( Value1, Value2, Value3),
(Value1, Value2, Value3)--//

INSERT INTO Employees (EmpID, EMP_Firstname, Salary, DOJ, Email,EMP_LastName)
Values(1, 'Bhaskar',50000,'2023-09-08','bhaskar@gmail.com','Shankeshi'),
(2,'Nagaraj',70000,'2025-09-07','nagaraj@gmail.com','N');



