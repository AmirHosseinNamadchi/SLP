SLP
===

### Successive Linear Programming (SLP) algorithm

In the SLP technique, the solution of the original nonlinear programming
problem is obtained by solving a series of linear programming problems.
Each linear programming problem is generated by linearizing the
nonlinear objective and constraint functions about the current design
vector. This is done using Taylor's theorem for multi-variable
functions.

### Problem Definition

SLP algorithm is employed for design of a hollow square-cross-section
cantilever beam to support a load of 20 (KN) at its end ([Adopted from
this book](http://www.sciencedirect.com/science/book/9780128008065)).
The beam is 2 (m) long. The failure conditions for the beam are as
follows:<br />
* the material should not fail under the action of the load
* the deflection of the free end should be no more than 1 (cm).

The width-to-thickness ratio for the beam should be no more than 8. A
minimum-mass beam is desired. The width and thickness of the beam must
be within the following limits: <br />60 (mm) &lt;= *w* &lt;=300 (mm)
<br />10 (mm) &lt;= *t* &lt;= 40 (mm)

### Dependencies

* numpy
* scipy

### About code
This code is mainly developed to solve the aforementioned problem.
Gradients of the objective and constraint functions is required and I've 
used finite difference approximations to evaluate them. In each step of SLP
a linear optimization problem must be solved. `linprog` was used for this purpose.


### Example
```python
pr = Problem()
opt = SLPOptimization()
x,f,viol = opt.run_SLP(pr,100,17)
```
Here, `pr` and `opt` is an instance of `Problem` and
`SLPOptimization` class, respectively. `run_SLP` method starts
SLP algorithm with initial guess `w = 100` and `t=17`. `x`, `f` and `viol`
represent optimal design vector, the corresponding cost function value and
violation amount.
