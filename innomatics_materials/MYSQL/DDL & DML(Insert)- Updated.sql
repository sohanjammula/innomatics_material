
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

Use Innomatics;
Drop table Employees;
Drop table Department;

CREATE TABLE Employee (
    EmpID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Department VARCHAR(50),
    Salary DECIMAL(10,2),
    HireDate DATE
);
INSERT INTO Employee (EmpID, FirstName, LastName, Department, Salary, HireDate) VALUES
(1, 'Rahul', 'Sharma', 'IT', 55000, '2020-01-15'),
(2, 'Sneha', 'Patel', 'HR', 45000, '2019-03-12'),
(3, 'Arjun', 'Mehta', 'Finance', 60000, '2021-07-01'),
(4, 'Priya', 'Reddy', 'IT', 52000, '2020-11-20'),
(5, 'Vikram', 'Singh', 'Sales', 48000, '2018-09-05'),
(6, 'Neha', 'Kapoor', 'Finance', 62000, '2022-02-10'),
(7, 'Kiran', 'Nair', 'HR', 43000, '2019-06-25'),
(8, 'Manoj', 'Joshi', 'Sales', 50000, '2021-04-18'),
(9, 'Divya', 'Iyer', 'IT', 57000, '2020-08-30'),
(10, 'Ravi', 'Kumar', 'Finance', 61000, '2017-12-01'),
(11, 'Anita', 'Desai', 'HR', 44000, '2018-05-14'),
(12, 'Suresh', 'Yadav', 'Sales', 49000, '2019-10-09'),
(13, 'Meena', 'Chopra', 'Finance', 63000, '2022-06-22'),
(14, 'Ajay', 'Verma', 'IT', 56000, '2021-01-11'),
(15, 'Pooja', 'Shah', 'HR', 46000, '2020-03-19'),
(16, 'Santosh', 'Rao', 'Sales', 51000, '2019-07-27'),
(17, 'Kavita', 'Bhat', 'Finance', 64000, '2021-09-15'),
(18, 'Deepak', 'Malhotra', 'IT', 58000, '2022-12-05'),
(19, 'Ramesh', 'Gupta', 'HR', 47000, '2018-11-21'),
(20, 'Sunita', 'Pillai', 'Sales', 52000, '2020-04-02'),
(21, 'Ashok', 'Menon', 'Finance', 65000, '2023-01-17'),
(22, 'Geeta', 'Kulkarni', 'IT', 59000, '2021-08-08');


CREATE TABLE Department (
    DeptID INT,
    DeptName VARCHAR(50) NOT NULL,
    Location VARCHAR(50)
);
INSERT INTO Department (DeptID, DeptName, Location) VALUES
(1, 'IT', 'Hyderabad'),
(2, 'HR', 'Mumbai'),
(3, 'Finance', 'Delhi'),
(4, 'Sales', 'Chennai'),
(5, 'Marketing', 'Bangalore'),
(6, 'Operations', 'Pune'),
(7, 'Support', 'Kolkata'),
(8, 'Research', 'Hyderabad'),
(9, 'Admin', 'Delhi'),
(10, 'Training', 'Chennai');

CREATE TABLE Project (
    ProjectID INT,
    ProjectName VARCHAR(100),
    DeptID INT,
    StartDate DATE,
    EndDate DATE
);
INSERT INTO Project (ProjectID, ProjectName, DeptID, StartDate, EndDate) VALUES
(101, 'ERP Implementation', 1, '2023-01-01', '2023-12-31'),
(102, 'Recruitment Drive', 2, '2023-03-01', '2023-06-30'),
(103, 'Budget Planning', 3, '2023-04-01', '2023-09-30'),
(104, 'Sales Campaign', 4, '2023-05-01', '2023-08-31'),
(105, 'Digital Marketing', 5, '2023-07-01', '2023-12-31');





