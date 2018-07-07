# Potts Model
This repository was built with the objective of studying the effects of dopage in the family of compounds <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\mathrm{CeCo}_{1-x}\mathrm{Fe}_{x}\mathrm{Si}^{}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\mathrm{CeCo}_{1-x}\mathrm{Fe}_{x}\mathrm{Si}^{}" title="\mathrm{CeCo}_{1-x}\mathrm{Fe}_{x}\mathrm{Si}^{}" /></a>
##  Random Number Generators

#### Tausworthe Generators 
Tausworthe Generator (TG) is a kind of multiplicative recursive generator which produces random bits. It has the following form:

<a href="https://www.codecogs.com/eqnedit.php?latex=x_{n&plus;1}&space;=&space;(A_{1}x_{n}&space;&plus;&space;A_{2}x_{n-1}&space;&plus;&space;\cdots&space;&plus;&space;A_{k}x_{n-k&plus;1})\mod{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{n&plus;1}&space;=&space;(A_{1}x_{n}&space;&plus;&space;A_{2}x_{n-1}&space;&plus;&space;\cdots&space;&plus;&space;A_{k}x_{n-k&plus;1})\mod{2}" title="x_{n+1} = (A_{1}x_{n} + A_{2}x_{n-1} + \cdots + A_{k}x_{n-k+1})\mod{2}" /></a>

where <img src="https://latex.codecogs.com/gif.latex?\inline&space;x_{i},&space;A_{i}\in&space;\left&space;\{&space;0,1&space;\right&space;\}\hspace{0.2cm}\forall&space;i" title="x_{i}, A_{i}\in \left \{ 0,1 \right \}\hspace{0.2cm}\forall i" />

The theory behind TG is related to irreducible primitive polynomials over GF(2). A polynomial over Galois field of order 2 (GF(2)) is a polynomial whose coefficients are either 0 and 1. Such a polynomial is irreducible primitive if it does not have nontrivial factors like 1 and has order of <img src="https://latex.codecogs.com/gif.latex?\inline&space;2^{n}-1" title="2^{n}-1" />
