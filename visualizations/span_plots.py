import numpy as np
import matplotlib.pyplot as plt


def plot_linear_combination(vector_a, vector_b, resultant):
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)
    resultant = np.asarray(resultant, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    for vector, label in (
        (vector_a, "Vector A"),
        (vector_b, "Vector B"),
        (resultant, "Linear Combination"),
    ):
        ax.quiver(0, 0, vector[0], vector[1], angles="xy", scale_units="xy", scale=1, label=label)

    all_values = np.concatenate((vector_a, vector_b, resultant))
    limit = max(2.0, float(np.max(np.abs(all_values))) * 1.3 + 0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear Combination")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
