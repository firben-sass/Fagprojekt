import numpy as np
import scipy.interpolate as interpolate

class BSpline:
    def __init__(self, knots_x, degree):
        self.knots_x = np.array(knots_x)
        self.degree = degree
        self.N = len(knots_x)

    def bspline_basis(self):
        K = self.degree + 1
        t = np.concatenate([[self.knots_x[0]]*K, self.knots_x, [self.knots_x[-1]]*K])
        degrees_of_freedom = self.N+K
        basis = interpolate.BSpline(t, np.eye(degrees_of_freedom), K, extrapolate=False)
        x_vals = np.linspace(self.knots_x.min(), self.knots_x.max(), 256)
        B = basis(x_vals)
        return B, x_vals