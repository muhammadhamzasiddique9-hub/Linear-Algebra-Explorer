# Linear Algebra Explorer

An interactive educational application for exploring core Linear Algebra concepts using Python, NumPy, Matplotlib, and Streamlit.

## Version 1.0 scope

This release covers the topics included in the project's current learning scope:

1. **Vector Explorer** — addition, subtraction, magnitude, dot product, and 2D visualization.
2. **Matrix Explorer** — matrix addition, multiplication, transpose, and inverse.
3. **Linear Combinations & Span** — calculate a linear combination and determine span dimension.
4. **Linear Independence** — analyze whether two vectors are linearly independent or dependent.
5. **Basis & Dimension** — calculate rank, pivot columns, column-space basis vectors, and nullity.
6. **Matrix Transformations** — apply a 2 × 2 matrix to a vector and visualize the transformation.
7. **Eigenvalues & Eigenvectors** — calculate eigenvalues, eigenvectors, characteristic polynomial, and diagonalization information.

This version intentionally stops at eigenvalues and eigenvectors. Later topics are not included in Version 1.0.

## Technologies

- Python 3.10 or newer
- NumPy
- Matplotlib
- Streamlit
- pytest (automated tests)

## Project structure

```text
Linear-Algebra-Explorer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── modules/
│   ├── validation.py
│   ├── vectors.py
│   ├── matrices.py
│   ├── linear_combinations.py
│   ├── independence.py
│   ├── basis_dimension.py
│   ├── transformations.py
│   └── eigenvalues.py
├── visualizations/
│   ├── vector_plots.py
│   ├── span_plots.py
│   ├── transformations_plots.py
│   └── eigen_plots.py
└── tests/
    └── ...
```

## Setup on Windows

Open the project folder in VS Code. In the VS Code terminal, create a virtual environment if you do not already have one:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the Python executable inside `.venv` directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal and normally open the application in your browser.

## Run automated tests

```powershell
python -m pytest -v
```

The tests check expected mathematical results, invalid inputs, matrix dimensions, singular matrices, and selected numerical edge cases.

## Numerical notes

- The application accepts real-valued user input. Eigenvalues and eigenvectors may be complex even when the input matrix is real.
- The eigenvector direction plot is a 2D visualization and only draws real eigenvector directions.
- Floating-point calculations are approximate. Very close-to-zero values can be affected by numerical tolerance.
- The basis explorer reports pivot columns from the original matrix; those original columns form a basis for its column space.

## Project goal

The goal is to make foundational Linear Algebra concepts easier to investigate through interactive numerical experiments and visualizations. The application is an educational aid; it does not replace mathematical proofs or a full Linear Algebra course.

## Future work

Future versions may add topics after the current learning scope, more visualization options, and deployment improvements. These are not part of Version 1.0.
