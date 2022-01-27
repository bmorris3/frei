Bezier interpolation
====================

Basic Bézier interpolation example:

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt

    from frei import evaluate_bezier

    points = np.random.random(size=(5, 2))

    interpolated = evaluate_bezier(points, n=10)

    plt.plot(points[:, 0], points[:, 1], '.r')
    plt.plot(interpolated[:, 0], interpolated[:, 1], 'b')
    plt.show()
