import numpy as np

__all__ = ['evaluate_bezier']


def get_bezier_coef(points):
    """
    Parameters
    ----------
    points : array with shape (n_points, 2)
        points to interpolate
        
    Returns
    -------
    A, B : arrays with shape (n_points - 1, 2)
        Solutions for the Bezier coefficients
    """
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = 2 * (2 * points[:-1] + points[1:])
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = np.zeros((n, 2))
    B[:-1] = 2 * points[1:-1] - A[1:]
    B[n - 1] = (A[n - 1] + points[n]) / 2
    return A, B


def get_cubic(a, b, c, d, t):
    return (
        np.power(1 - t, 3) * a + 
        3 * np.power(1 - t, 2) * t * b + 
        3 * (1 - t) * np.power(t, 2) * c + 
        np.power(t, 3) * d
    )


def get_bezier_cubic(points, t):
    A, B = get_bezier_coef(points)
    broadcast = np.ones_like(t)
    pts_broadcast = points[None, :-1, :] * broadcast
    pts_p1_broadcast = points[None, 1:, :] * broadcast
    return get_cubic(
        pts_broadcast, A, B, 
        pts_p1_broadcast, t
    ).reshape((-1, 2), order='F')


def evaluate_bezier(points, n):
    """
    Parameters
    ----------
    points : array with shape (n_points, 2)
        points to interpolate
    n : int
        Number of interpolated points to return between neighboring points
    
    Returns
    -------
    A, B : arrays with shape (n_points - 1, 2)
        Solutions for the Bezier coefficients
    """ 
    t = np.linspace(0, 1, n)[:, None, None]
    curves = get_bezier_cubic(points, t)
    return curves
