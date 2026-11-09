import numpy as np
import matplotlib.pyplot as plt


def plot_eigenvectors(matrix, eigenvalues, eigenvectors):
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues = np.asarray(eigenvalues)
    eigenvectors = np.asarray(eigenvectors)

    fig, ax = plt.subplots(figsize=(8, 7))
    real_directions = []
    for index, eigenvalue in enumerate(eigenvalues):
        eigenvector = eigenvectors[:, index]
        if abs(complex(eigenvalue).imag) > 1e-10:
            continue
        if not np.all(np.abs(eigenvector.imag) <= 1e-10):
            continue
        vector = np.real(eigenvector).astype(float)
        magnitude = float(np.linalg.norm(vector))
        if np.isclose(magnitude, 0.0):
            continue
        real_directions.append((index, vector / magnitude))

    limit = 5.0
    if real_directions:
        max_component = max(float(np.max(np.abs(v))) for _, v in real_directions)
        limit = max(2.0, 2.5 * max_component)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.25)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Real Eigenvector Directions")

    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for index, vector in real_directions:
        color = colors[index % len(colors)]
        ax.plot(
            [-limit * vector[0], limit * vector[0]],
            [-limit * vector[1], limit * vector[1]],
            color=color,
            alpha=0.65,
            label=f"Eigenvector {index + 1}",
        )
        ax.quiver(0, 0, vector[0], vector[1], angles="xy", scale_units="xy", scale=1, color=color)

    if real_directions:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No real eigenvector directions", ha="center", va="center", transform=ax.transAxes)

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
