import numpy as np
from scipy import integrate, interpolate
from numpy.polynomial.legendre import leggauss
from scipy.special import erf

class IntegrationMethods:
    @staticmethod
    def trapezoid(f, a, b, n=1000):
        x = np.linspace(a, b, n+1)
        y = np.vectorize(f)(x)
        return np.trapz(y, x)

    @staticmethod
    def simpson(f, a, b, n=1000):
        if n % 2 != 0: n += 1
        x = np.linspace(a, b, n+1)
        y = np.vectorize(f)(x)
        return integrate.simpson(y, x=x)

    @staticmethod
    def boole(f, a, b, n=1000):
        n = max(4, (n // 4) * 4)
        x = np.linspace(a, b, n+1)
        y = np.vectorize(f)(x)
        h = (b - a) / n
        s = 7*y[0] + 7*y[-1]
        s += 32*np.sum(y[1:-1:4]) + 32*np.sum(y[3:-1:4])
        s += 12*np.sum(y[2:-1:4])
        s += 14*np.sum(y[4:-1:4])
        return 2*h*s/45

    @staticmethod
    def romberg(f, a, b, n=None):
        return integrate.romberg(f, a, b, tol=1e-10, divmax=20)

    @staticmethod
    def gaussian_quadrature(f, a, b, n=100):
        pts, wts = leggauss(n)
        t = 0.5*(b - a)*pts + 0.5*(b + a)
        return 0.5*(b - a) * np.dot(wts, np.vectorize(f)(t))

    @staticmethod
    def adaptive_simpson(f, a, b, n=None):
        return integrate.quad(f, a, b, limit=200, epsabs=1e-10, epsrel=1e-10)[0]

    @staticmethod
    def cubic_spline(f, a, b, n=1000):
        x = np.linspace(a, b, n+1)
        y = np.vectorize(f)(x)
        cs = interpolate.CubicSpline(x, y)
        return float(cs.integrate(a, b))

    @staticmethod
    def spectral(f, a, b, n=512):
        k = n
        theta = np.pi * np.arange(k+1) / k
        x_cc = np.cos(theta)
        x_ab = 0.5*(b - a)*x_cc + 0.5*(b + a)
        y = np.vectorize(f)(x_ab)
        c = np.fft.rfft(y).real
        c[1:-1] *= 2
        j = np.arange(len(c))
        w_j = np.where(j % 2 == 0, 2.0/(1 - j**2 + (j==0)*1e-30), 0.0)
        w_j[0] = 2.0
        return (np.dot(w_j, c) / k) * 0.5*(b - a)

    @staticmethod
    def bayesian_quadrature(f, a, b, n=30):
        x = np.sort(np.random.uniform(a, b, n))
        y = np.vectorize(f)(x)
        ell = (b - a) / 5.0
        sigma2 = np.var(y) + 1e-6
        def kernel(x1, x2):
            return sigma2 * np.exp(-0.5 * ((x1 - x2) / ell)**2)
        K = kernel(x[:, None], x[None, :]) + 1e-8 * np.eye(n)
        def kernel_mean(xi):
            return sigma2 * ell * np.sqrt(np.pi) * 0.5 * (
                erf((b - xi)/(np.sqrt(2)*ell)) - erf((a - xi)/(np.sqrt(2)*ell))
            )
        z = np.array([kernel_mean(xi) for xi in x])
        alpha = np.linalg.solve(K, y)
        return float(np.dot(z, alpha))

    @staticmethod
    def tanh_sinh(f, a, b, n=None, h=0.05):
        t_max = 3.5
        ts = np.arange(-t_max, t_max + h, h)
        phi = np.tanh(0.5*np.pi*np.sinh(ts))
        dphi = 0.5*np.pi*np.cosh(ts) / np.cosh(0.5*np.pi*np.sinh(ts))**2
        x_ab = 0.5*(b - a)*phi + 0.5*(b + a)
        weights = dphi * 0.5*(b - a)
        mask = (x_ab > a + 1e-14) & (x_ab < b - 1e-14)
        x_ab = x_ab[mask]; weights = weights[mask]
        y = np.array([f(xi) for xi in x_ab], dtype=float)
        y[~np.isfinite(y)] = 0.0
        return float(h * np.dot(weights, y))

METHODS = {
    "Trapezoid":         IntegrationMethods.trapezoid,
    "Simpson":           IntegrationMethods.simpson,
    "Boole":             IntegrationMethods.boole,
    "Romberg":           IntegrationMethods.romberg,
    "Gaussian":          IntegrationMethods.gaussian_quadrature,
    "Adaptive Simpson":  IntegrationMethods.adaptive_simpson,
    "Cubic Spline":      IntegrationMethods.cubic_spline,
    "Spectral":          IntegrationMethods.spectral,
    "Bayesian QD":       IntegrationMethods.bayesian_quadrature,
    "Tanh-Sinh":         IntegrationMethods.tanh_sinh,
}
