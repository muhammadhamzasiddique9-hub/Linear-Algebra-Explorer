"""Streamlit interface for the Linear Algebra Explorer, Version 1.0."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from modules.vectors import add_vectors, subtract_vectors, vector_magnitude, dot_product
from modules.matrices import add_matrices, multiply_matrices, transpose_matrix, inverse_matrix
from modules.linear_combinations import (
    calculate_linear_combination,
    calculate_span_dimension,
    describe_span,
)
from modules.independence import analyze_independence, describe_independence
from modules.basis_dimension import analyze_basis, describe_basis
from modules.transformations import apply_transformation, analyze_transformation
from modules.eigenvalues import analyze_eigenvalues
from visualizations.vector_plots import plot_vectors
from visualizations.span_plots import plot_linear_combination
from visualizations.transformations_plots import plot_transformation
from visualizations.eigen_plots import plot_eigenvectors


# -----------------------------------------------------------------------------
# Page setup and shared helpers
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Linear Algebra Explorer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def create_vector_input(label, prefix, default_x=1.0, default_y=2.0):
    """Create a two-dimensional vector input with stable Streamlit keys."""
    left, right = st.columns(2)
    with left:
        x_value = st.number_input(
            f"{label} — x component",
            value=float(default_x),
            step=0.5,
            format="%.4f",
            key=f"{prefix}_x",
        )
    with right:
        y_value = st.number_input(
            f"{label} — y component",
            value=float(default_y),
            step=0.5,
            format="%.4f",
            key=f"{prefix}_y",
        )
    return np.array([x_value, y_value], dtype=float)


def create_matrix_input(name, rows, columns, prefix):
    """Create a matrix input. The default is an identity-shaped matrix."""
    values = []
    for row_index in range(rows):
        row_values = []
        input_columns = st.columns(columns)
        for column_index in range(columns):
            default = 1.0 if row_index == column_index else 0.0
            with input_columns[column_index]:
                value = st.number_input(
                    f"{name} [{row_index + 1}, {column_index + 1}]",
                    value=default,
                    step=0.5,
                    format="%.4f",
                    key=f"{prefix}_{row_index}_{column_index}",
                )
            row_values.append(value)
        values.append(row_values)
    return np.asarray(values, dtype=float)


def display_complex_number(value, digits=4):
    """Format a real or complex scalar without displaying unnecessary parts."""
    number = complex(value)
    real = 0.0 if np.isclose(number.real, 0.0) else number.real
    imag = 0.0 if np.isclose(number.imag, 0.0) else number.imag
    if imag == 0.0:
        return f"{real:.{digits}f}"
    if real == 0.0:
        return f"{imag:.{digits}f}i"
    sign = "+" if imag > 0 else "−"
    return f"{real:.{digits}f} {sign} {abs(imag):.{digits}f}i"


def display_polynomial(coefficients):
    """Format descending polynomial coefficients using lambda notation."""
    degree = len(coefficients) - 1
    terms = []
    for index, coefficient in enumerate(coefficients):
        value = float(np.real_if_close(coefficient))
        power = degree - index
        if np.isclose(value, 0.0):
            continue
        magnitude = abs(value)
        if power == 0:
            body = f"{magnitude:.4g}"
        elif power == 1:
            body = "λ" if np.isclose(magnitude, 1.0) else f"{magnitude:.4g}λ"
        else:
            body = f"λ^{power}" if np.isclose(magnitude, 1.0) else f"{magnitude:.4g}λ^{power}"
        if not terms:
            terms.append(body if value > 0 else f"−{body}")
        else:
            terms.append(("+ " if value > 0 else "− ") + body)
    return " ".join(terms) if terms else "0"


def show_figure(figure):
    """Display and close a Matplotlib figure to avoid resource accumulation."""
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def show_intro(title, description, formula=None):
    st.header(title)
    st.write(description)
    if formula:
        st.latex(formula)


# -----------------------------------------------------------------------------
# Header and navigation
# -----------------------------------------------------------------------------

st.title("📐 Linear Algebra Explorer")
st.write(
    "An interactive learning application for exploring vectors, matrices, "
    "linear combinations, independence, basis, transformations, and eigenvalues."
)
st.caption("Version 1.0 · Python · NumPy · Matplotlib · Streamlit")

with st.sidebar:
    st.title("Explorer Menu")
    st.write("Choose a topic from the tabs across the top of the application.")
    st.divider()
    st.markdown("**Included topics**")
    st.markdown(
        "1. Vectors\n"
        "2. Matrices\n"
        "3. Linear combinations and span\n"
        "4. Linear independence\n"
        "5. Basis and dimension\n"
        "6. Matrix transformations\n"
        "7. Eigenvalues and eigenvectors"
    )
    st.divider()
    st.caption("Educational tool: numerical results are computed using NumPy.")


tabs = st.tabs(
    [
        "Vectors",
        "Matrices",
        "Combinations & Span",
        "Independence",
        "Basis & Dimension",
        "Transformations",
        "Eigenvalues & Eigenvectors",
    ]
)


# -----------------------------------------------------------------------------
# 1. Vector Explorer
# -----------------------------------------------------------------------------

with tabs[0]:
    show_intro(
        "Vector Explorer",
        "Perform basic operations on two-dimensional vectors and view their geometric meaning.",
        r"\mathbf{a}+\mathbf{b},\quad \mathbf{a}-\mathbf{b},\quad "
        r"\|\mathbf{a}\|=\sqrt{a_1^2+a_2^2},\quad "
        r"\mathbf{a}\cdot\mathbf{b}=a_1b_1+a_2b_2",
    )
    vector_a = create_vector_input("Vector A", "vector_a", 3.0, 2.0)
    vector_b = create_vector_input("Vector B", "vector_b", 1.0, 3.0)

    if st.button("Calculate vector operations", type="primary", key="calculate_vectors"):
        try:
            addition = add_vectors(vector_a, vector_b)
            subtraction = subtract_vectors(vector_a, vector_b)
            magnitude_a = vector_magnitude(vector_a)
            magnitude_b = vector_magnitude(vector_b)
            product = dot_product(vector_a, vector_b)

            result_columns = st.columns(5)
            result_columns[0].metric("A + B", np.array2string(addition, precision=3))
            result_columns[1].metric("A − B", np.array2string(subtraction, precision=3))
            result_columns[2].metric("‖A‖", f"{magnitude_a:.4f}")
            result_columns[3].metric("‖B‖", f"{magnitude_b:.4f}")
            result_columns[4].metric("A · B", f"{product:.4f}")
            st.subheader("Geometric visualization")
            show_figure(plot_vectors(vector_a, vector_b))
        except (ValueError, np.linalg.LinAlgError) as error:
            st.error(f"Could not calculate the vector operations: {error}")


# -----------------------------------------------------------------------------
# 2. Matrix Explorer
# -----------------------------------------------------------------------------

with tabs[1]:
    show_intro(
        "Matrix Explorer",
        "Explore matrix addition, multiplication, transpose, and inverse for 2 × 2 matrices.",
        r"A+B,\quad AB,\quad A^T,\quad A^{-1}\;\text{where}\;AA^{-1}=I",
    )
    st.caption("For multiplication, the number of columns in A must equal the number of rows in B.")
    left, right = st.columns(2)
    with left:
        st.subheader("Matrix A")
        matrix_a = create_matrix_input("A", 2, 2, "matrix_a")
    with right:
        st.subheader("Matrix B")
        matrix_b = create_matrix_input("B", 2, 2, "matrix_b")

    if st.button("Calculate matrix operations", type="primary", key="calculate_matrices"):
        try:
            st.subheader("Addition: A + B")
            st.write(add_matrices(matrix_a, matrix_b))
            st.subheader("Multiplication: AB")
            st.write(multiply_matrices(matrix_a, matrix_b))
            st.subheader("Transpose: Aᵀ")
            st.write(transpose_matrix(matrix_a))
            st.subheader("Inverse: A⁻¹")
            try:
                st.write(inverse_matrix(matrix_a))
            except np.linalg.LinAlgError:
                st.warning("Matrix A is singular, so its inverse does not exist.")
        except ValueError as error:
            st.error(f"Could not calculate the matrix operations: {error}")


# -----------------------------------------------------------------------------
# 3. Linear Combinations and Span
# -----------------------------------------------------------------------------

with tabs[2]:
    show_intro(
        "Linear Combinations & Span",
        "Change the coefficients to see how two vectors combine and determine the dimension of their span.",
        r"c_1\mathbf{a}+c_2\mathbf{b}",
    )
    vector_a = create_vector_input("Vector A", "linear_a", 1.0, 0.0)
    vector_b = create_vector_input("Vector B", "linear_b", 0.0, 1.0)
    first, second = st.columns(2)
    with first:
        coefficient_a = st.number_input("Coefficient c₁", value=1.0, step=0.5, key="coefficient_a")
    with second:
        coefficient_b = st.number_input("Coefficient c₂", value=1.0, step=0.5, key="coefficient_b")

    if st.button("Calculate linear combination", type="primary", key="calculate_combination"):
        try:
            result = calculate_linear_combination(vector_a, vector_b, coefficient_a, coefficient_b)
            dimension = calculate_span_dimension(vector_a, vector_b)
            st.subheader("Result")
            st.write(result)
            st.metric("Dimension of span", dimension)
            st.info(describe_span(dimension))
            show_figure(plot_linear_combination(vector_a, vector_b, result))
        except ValueError as error:
            st.error(f"Could not calculate the linear combination: {error}")


# -----------------------------------------------------------------------------
# 4. Linear Independence
# -----------------------------------------------------------------------------

with tabs[3]:
    show_intro(
        "Linear Independence",
        "Check whether either vector can be represented as a scalar multiple of the other.",
        r"c_1\mathbf{a}+c_2\mathbf{b}=\mathbf{0}",
    )
    vector_a = create_vector_input("Vector A", "independence_a", 1.0, 0.0)
    vector_b = create_vector_input("Vector B", "independence_b", 0.0, 1.0)

    if st.button("Analyze independence", type="primary", key="analyze_independence"):
        try:
            result = analyze_independence(vector_a, vector_b)
            first, second = st.columns(2)
            first.metric("Rank", result["rank"])
            if result["determinant"] is not None:
                second.metric("Determinant", f"{result['determinant']:.5f}")
            if result["independent"]:
                st.success("The vectors are linearly independent.")
            else:
                st.warning("The vectors are linearly dependent.")
            st.info(describe_independence(result["rank"]))
        except ValueError as error:
            st.error(f"Could not analyze independence: {error}")


# -----------------------------------------------------------------------------
# 5. Basis and Dimension
# -----------------------------------------------------------------------------

with tabs[4]:
    show_intro(
        "Basis & Dimension",
        "Find the rank, pivot columns, a basis for the column space, and the nullity of a 3 × 3 matrix.",
        r"\text{Rank}(A)+\text{Nullity}(A)=\text{number of columns of }A",
    )
    matrix = create_matrix_input("Basis matrix", 3, 3, "basis_matrix")

    if st.button("Analyze basis and dimension", type="primary", key="analyze_basis"):
        try:
            result = analyze_basis(matrix)
            metrics = st.columns(3)
            metrics[0].metric("Rank", result["rank"])
            metrics[1].metric("Dimension", result["dimension"])
            metrics[2].metric("Nullity", result["nullity"])
            st.subheader("Pivot columns")
            if result["pivot_columns"]:
                st.write([column + 1 for column in result["pivot_columns"]])
            else:
                st.write("No pivot columns")
            st.subheader("Basis vectors for the column space")
            if result["basis_vectors"]:
                for index, basis_vector in enumerate(result["basis_vectors"], start=1):
                    st.write(f"Basis vector {index}", basis_vector)
            else:
                st.write("The column space contains only the zero vector.")
            st.info(describe_basis(result["rank"], matrix.shape[0], matrix.shape[1]))
        except ValueError as error:
            st.error(f"Could not analyze the matrix: {error}")


# -----------------------------------------------------------------------------
# 6. Matrix Transformations
# -----------------------------------------------------------------------------

with tabs[5]:
    show_intro(
        "Matrix Transformations",
        "Apply a 2 × 2 matrix to a vector and explore how the matrix changes the plane.",
        r"T(\mathbf{x})=A\mathbf{x},\quad \text{area scale}=|\det(A)|",
    )
    matrix = create_matrix_input("Transformation matrix", 2, 2, "transformation_matrix")
    vector = create_vector_input("Input vector", "transformation_vector", 2.0, 1.0)

    if st.button("Apply transformation", type="primary", key="apply_transformation"):
        try:
            transformed = apply_transformation(matrix, vector)
            analysis = analyze_transformation(matrix)
            first, second, third = st.columns(3)
            first.metric("Transformed vector", np.array2string(transformed, precision=3))
            second.metric("Determinant", f"{analysis['determinant']:.5f}")
            third.metric("Area scale factor", f"{analysis['area_scale_factor']:.5f}")
            if analysis["singular"]:
                st.warning(analysis["description"])
            else:
                st.success(analysis["description"])
            show_figure(plot_transformation(matrix, vector))
        except ValueError as error:
            st.error(f"Could not apply the transformation: {error}")


# -----------------------------------------------------------------------------
# 7. Eigenvalues and Eigenvectors
# -----------------------------------------------------------------------------

with tabs[6]:
    show_intro(
        "Eigenvalues & Eigenvectors",
        "Analyze a real-valued square matrix. Eigenvalues and eigenvectors may be real or complex.",
        r"A\mathbf{v}=\lambda\mathbf{v},\quad \det(\lambda I-A)=0,\quad A=PDP^{-1}",
    )
    st.caption("Choose a matrix size. The direction plot is available for 2 × 2 matrices with real eigenvector directions.")
    size = st.selectbox("Matrix size", [2, 3, 4], index=0, key="eigen_matrix_size")
    matrix = create_matrix_input("Eigenvalue matrix", size, size, "eigen_matrix")

    if st.button("Analyze eigenvalues and eigenvectors", type="primary", key="analyze_eigenvalues"):
        try:
            result = analyze_eigenvalues(matrix)
            eigenvalues = result["eigenvalues"]
            eigenvectors = result["eigenvectors"]
            diagonalization = result["diagonalization"]

            st.subheader("1. Eigenvalues")
            eigenvalue_columns = st.columns(size)
            for index, value in enumerate(eigenvalues):
                eigenvalue_columns[index].metric(f"λ{index + 1}", display_complex_number(value))

            st.subheader("2. Eigenvectors")
            for index in range(size):
                with st.expander(f"Eigenvector {index + 1}"):
                    st.write(np.real_if_close(eigenvectors[:, index]))

            st.subheader("3. Characteristic polynomial")
            st.latex(r"\det(\lambda I-A)=0")
            polynomial = display_polynomial(result["characteristic_coefficients"])
            st.write(f"{polynomial} = 0")
            st.caption("The polynomial is calculated from det(λI − A).")

            st.subheader("4. Diagonalization")
            st.metric("Rank of eigenvector matrix", diagonalization["rank"])
            if diagonalization["diagonalizable"]:
                st.success("The matrix is diagonalizable over the complex numbers.")
                first, second = st.columns(2)
                with first:
                    st.write("Diagonal matrix D")
                    st.write(np.real_if_close(diagonalization["diagonal_matrix"]))
                with second:
                    st.write("Inverse eigenvector matrix P⁻¹")
                    st.write(np.real_if_close(diagonalization["inverse_eigenvector_matrix"]))
            else:
                st.warning("The matrix is not diagonalizable over the complex numbers.")

            if diagonalization["real_diagonalizable"]:
                st.success("The matrix is also diagonalizable over the real numbers.")
            else:
                st.info("The matrix does not have a complete real eigenvector basis.")

            if size == 2:
                st.subheader("5. Eigenvector directions")
                show_figure(plot_eigenvectors(matrix, eigenvalues, eigenvectors))
                if np.any(np.abs(eigenvalues.imag) > 1e-10):
                    st.caption("Complex eigenvalues do not represent real directions in this two-dimensional plot.")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            st.error(f"Unable to analyze this matrix: {error}")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.divider()
st.caption("Linear Algebra Explorer · Version 1.0 · Built for learning and experimentation")
