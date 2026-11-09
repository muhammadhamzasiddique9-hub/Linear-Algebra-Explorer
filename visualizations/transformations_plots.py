import numpy as np
import matplotlib.pyplot as plt


def plot_transformation(matrix, vector):
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 8))

    base = np.linspace(-5, 5, 21)
    for value in base:
        ax.plot([-5, 5], [value, value], color="gray", alpha=0.22, linewidth=0.7)
        ax.plot([value, value], [-5, 5], color="gray", alpha=0.22, linewidth=0.7)

    transformed_points = []
    for value in np.linspace(-5, 5, 11):
        vertical = np.array([[value, -5.0], [value, 5.0]])
        horizontal = np.array([[-5.0, value], [5.0, value]])
        tv = (matrix @ vertical.T).T
        th = (matrix @ horizontal.T).T
        transformed_points.extend((tv, th))
        ax.plot(tv[:, 0], tv[:, 1], color="tab:blue", alpha=0.35, linewidth=0.8)
        ax.plot(th[:, 0], th[:, 1], color="tab:blue", alpha=0.35, linewidth=0.8)

    e1 = matrix @ np.array([1.0, 0.0])
    e2 = matrix @ np.array([0.0, 1.0])
    result = matrix @ vector
    vectors = (
        (e1, "Transformed e₁", "tab:blue"),
        (e2, "Transformed e₂", "tab:purple"),
        (vector, "Original vector", "tab:green"),
        (result, "Transformed vector", "tab:red"),
    )
    for item, label, color in vectors:
        ax.quiver(0, 0, item[0], item[1], angles="xy", scale_units="xy", scale=1, color=color, label=label)

    corners = np.array([[-5, -5], [-5, 5], [5, -5], [5, 5]], dtype=float)
    transformed_corners = (matrix @ corners.T).T
    all_points = np.vstack([corners, transformed_corners, vector, result, e1, e2])
    limit = max(5.0, float(np.max(np.abs(all_points))) * 1.12 + 0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Matrix Transformation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    return fig
