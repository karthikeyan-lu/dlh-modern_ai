# Machine Learning Foundations - AI Academy (DLH)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.25.2-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.3-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![SQL](https://img.shields.io/badge/SQL-MySQL-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Shell%20%26%20PyMongo-green)
![Calculus](https://img.shields.io/badge/Calculus-Core-red)
![Linear Algebra](https://img.shields.io/badge/Linear%20Algebra-Advanced-purple)
![Probability](https://img.shields.io/badge/Probability-Distributions-orange)
![Bayesian Probability](https://img.shields.io/badge/Bayesian%20Probability-Inference-teal)
![Multivariate Probability](https://img.shields.io/badge/Multivariate%20Probability-Gaussian%20Models-green)
![Statistics](https://img.shields.io/badge/Statistics-Foundations-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Level](https://img.shields.io/badge/Level-Beginner%20to%20Advanced-red)

This repository contains my mathematics, probability, visualization, and data pipeline exercises completed during the AI Academy program at Digital Learning Hub Luxembourg (DLH). The projects provide hands-on experience with calculus, linear algebra, probability, Bayesian inference, multivariate probability, statistics, data visualization, Pandas, SQL, MongoDB, and foundational machine learning mathematics.

---

## Objective

To build the mathematical and technical foundations required for machine learning by learning:

- Calculus notation and polynomial calculus
- Linear algebra operations using Python and NumPy
- Advanced matrix operations and definiteness
- Probability distributions and statistical modeling
- Bayesian probability and posterior inference
- Multivariate probability and Gaussian modeling
- Data visualization using Matplotlib
- Data manipulation and analysis using Pandas
- SQL database querying and optimization
- MongoDB shell operations
- PyMongo document workflows
- Data pipeline and log analytics concepts

---

## Topics Covered

### Mathematics

- Summation and product notation
- Derivatives and partial derivatives
- Indefinite, definite, and double integrals
- Matrix shapes and slicing
- Matrix addition, concatenation, and multiplication
- Determinants, minors, cofactors, adjugates, and inverses
- Eigenvalues and matrix definiteness

### Probability and Statistics

- Poisson Distribution
- Exponential Distribution
- Normal Distribution
- Binomial Distribution
- PMF (Probability Mass Function)
- PDF (Probability Density Function)
- CDF (Cumulative Distribution Function)
- Mean and Variance
- Standard Deviation
- Z-Scores
- Method of Moments
- Prior Probability
- Likelihood
- Marginal Probability
- Posterior Probability
- Continuous Posterior Intervals
- Mean Vectors
- Covariance Matrices
- Correlation Matrices
- Multivariate Normal Distribution
- Multivariate PDF

### Visualization

- Line graphs
- Scatter plots
- Histograms
- Stacked bar charts
- Subplots
- Logarithmic scales
- Color gradients and color bars
- PCA visualization in 3D

### Data Analysis

- Pandas DataFrames
- Data cleaning and preprocessing
- Missing value handling
- Time-series analysis
- Hierarchical indexing
- Data aggregation and visualization

### Data Pipeline and Databases

- MySQL database and table creation
- SQL filtering, sorting, aggregation, and joins
- SQL constraints, indexes, views, triggers, functions, and procedures
- MongoDB CRUD operations
- MongoDB Aggregation Framework
- PyMongo document workflows
- Nginx log analytics

---

## Project Modules

| Folder | Description |
| --- | --- |
| `math/calculus` | Calculus notation, derivatives, integrals, and polynomial calculus |
| `math/linear_algebra` | Matrix operations using Python lists and NumPy |
| `math/advanced_linear_algebra` | Determinants, inverses, adjugates, minors, cofactors, and definiteness |
| `math/probability` | Probability distributions and statistical modeling |
| `math/bayesian_prob` | Bayesian probability, likelihood, marginal probability, and posterior inference |
| `math/multivariate_prob` | Mean vectors, covariance, correlation, and multivariate normal distributions |
| `math/plotting` | Data visualization using Matplotlib and NumPy |
| `pipeline/pandas` | Data manipulation, cleaning, analysis, and visualization using Pandas |
| `pipeline/databases` | MySQL, MongoDB shell, and PyMongo exercises |

---

## Key Concepts Used

### Mathematics

- Calculus
- Linear Algebra
- Matrix Theory
- Eigenvalues and Definiteness

### Probability and Statistics

- Probability Distributions
- PMF, PDF, and CDF
- Statistical Estimation
- Bayesian Inference
- Prior, Likelihood, Marginal, and Posterior Probability
- Multivariate Probability
- Covariance and Correlation
- Multivariate Normal Distributions
- Mean and Variance
- Standard Deviation
- Z-Scores
- Method of Moments

### Programming

- Python Functions
- Recursive Algorithms
- Object-Oriented Programming
- Numerical Computing

### Data Technologies

- NumPy
- Pandas
- Matplotlib
- SQL
- MySQL
- MongoDB
- PyMongo

---

## Project Structure

```text
dlh-machine_learning/
|
├── math/
│   ├── calculus/
│   ├── linear_algebra/
│   ├── advanced_linear_algebra/
│   ├── probability/
│   ├── bayesian_prob/
│   ├── multivariate_prob/
│   └── plotting/
│
├── pipeline/
│   ├── pandas/
│   └── databases/
│
├── requirements.txt
├── aiacademy.yml
└── README.md
```

---

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- Matplotlib 3.8.3
- Pandas
- SciPy
- Pillow 10.3.0
- MySQL
- MongoDB
- PyMongo
- `pycodestyle` 2.11.1

---

## Usage

Run a linear algebra task:

```bash
python3 math/linear_algebra/0-slice_me_up.py
```

Run a probability task:

```bash
python3 math/probability/normal.py
```

Run a Bayesian probability task:

```bash
python3 math/bayesian_prob/3-main.py
```

Run a multivariate probability task:

```bash
python3 math/multivariate_prob/3-main.py
```

Run a plotting task:

```bash
python3 math/plotting/6-bars.py
```

Run a Pandas task:

```bash
python3 pipeline/pandas/14-visualize.py
```

Run a MySQL script:

```bash
mysql -uroot -p < pipeline/databases/0-create_database_if_missing.sql
```

Run a PyMongo script:

```bash
python3 pipeline/databases/34-log_stats.py
```

---

## Module READMEs

- `math/calculus/README.md`
- `math/linear_algebra/README.md`
- `math/advanced_linear_algebra/README.md`
- `math/probability/README.md`
- `math/bayesian_prob/README.md`
- `math/multivariate_prob/README.md`
- `math/plotting/README.md`
- `pipeline/pandas/README.md`
- `pipeline/databases/README.md`

---

## Author

Karthikeyan Marimuthu - AI Academy, Digital Learning Hub Luxembourg