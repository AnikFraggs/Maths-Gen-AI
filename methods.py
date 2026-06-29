from dataset import DATASET
from methods import METHODS
class IntegrationMethods:
    """
    All methods accept (f, a, b, n_or_options) and return a float.
    n = number of sub-intervals (or points, depending on method).
    """
 
    # ── 1. Trapezoid Rule ─────────────────────────────────────────
    @staticmethod
    def trapezoid(f, a, b, n=1000):
        x = np.linspace(a, b, n+1)
        y = f(x)
        return np.trapz(y, x)
 
    # ── 2. Simpson's 1/3 Rule ────────────────────────────────────
    @staticmethod
    def simpson(f, a, b, n=1000):
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
 
    # ── 3. Boole's Rule ──────────────────────────────────────────
    @staticmethod
    def boole(f, a, b, n=1000):
        # n must be multiple of 4
        n = max(4, (n // 4) * 4)
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        result = 0.0
        for i in range(0, n, 4):
            result += (2*h/45) * (
                7*y[i] + 32*y[i+1] + 12*y[i+2] + 32*y[i+3] + 7*y[i+4]
            )
        return result
 
    # ── 4. Romberg Integration ───────────────────────────────────
    @staticmethod
    def romberg(f, a, b, n=None):
        # n unused; uses scipy.integrate.romberg
        return integrate.romberg(f, a, b, tol=1e-10, divmax=20, show=False)
 
    # ── 5. Gaussian Quadrature ───────────────────────────────────
    @staticmethod
    def gaussian_quadrature(f, a, b, n=100):
        pts, wts = leggauss(n)
        # Transform from [-1,1] → [a,b]
        t = 0.5*(b - a)*pts + 0.5*(b + a)
        return 0.5*(b - a) * np.dot(wts, f(t))
 
    # ── 6. Adaptive Simpson ───────────────────────────────────────
    @staticmethod
    def adaptive_simpson(f, a, b, n=None, tol=1e-10):
        result, _ = integrate.quad(f, a, b, limit=200, epsabs=tol, epsrel=tol)
        return result
 
    # ── 7. Cubic Spline ──────────────────────────────────────────
    @staticmethod
    def cubic_spline(f, a, b, n=1000):
        x = np.linspace(a, b, n+1)
        y = f(x)
        cs = interpolate.CubicSpline(x, y)
        return float(cs.integrate(a, b))
 
    # ── 8. Spectral (Clenshaw-Curtis) ─────────────────────────────
    @staticmethod
    def spectral(f, a, b, n=512):
        # Clenshaw–Curtis: Chebyshev nodes + DCT-based weights
        k = n
        theta = np.pi * np.arange(k+1) / k
        x_cc = np.cos(theta)                        # in [-1, 1]
        x_ab = 0.5*(b - a)*x_cc + 0.5*(b + a)      # mapped to [a,b]
        y = f(x_ab)
        # Compute weights via FFT
        c = np.fft.rfft(y)                          # cosine coefficients
        c = c.real
        c[1:-1] *= 2                                # interior terms
        j = np.arange(len(c))
        # Weight = integral of T_j on [-1,1]
        w_j = np.where(j % 2 == 0, 2.0/(1 - j**2 + (j==0)*1e-30), 0.0)
        w_j[0] = 2.0                                # j=0 term
        # avoid div-by-zero for j=1 (odd, w=0 anyway)
        integral_11 = np.dot(w_j, c) / k
        return integral_11 * 0.5*(b - a)
 
    # ── 9. Bayesian Quadrature (GP-based, lite version) ──────────
    @staticmethod
    def bayesian_quadrature(f, a, b, n=30):
        """
        Lite Bayesian Quadrature with squared-exponential kernel.
        Chooses n points by Latin Hypercube, fits a GP, then
        computes E[∫f] analytically under the GP prior.
        """
        # Sample points
        x = np.sort(np.random.uniform(a, b, n))
        y = f(x)
 
        # Kernel: squared exponential
        ell = (b - a) / 5.0          # length scale
        sigma2 = np.var(y) + 1e-6    # signal variance
 
        def kernel(x1, x2):
            return sigma2 * np.exp(-0.5 * ((x1 - x2) / ell)**2)
 
        K = np.array([[kernel(xi, xj) for xj in x] for xi in x])
        K += 1e-8 * np.eye(n)        # jitter
 
        # Kernel mean (integral of k(·, x) over [a,b])
        def kernel_mean(xi):
            return sigma2 * ell * np.sqrt(2*np.pi/2) * (
                __import__('scipy').special.erf((b - xi)/(np.sqrt(2)*ell)) -
                __import__('scipy').special.erf((a - xi)/(np.sqrt(2)*ell))
            )
 
        z = np.array([kernel_mean(xi) for xi in x])
        alpha = np.linalg.solve(K, y)
        return float(np.dot(z, alpha))
 
    # ── 10. Tanh-Sinh (Double Exponential) ───────────────────────
    @staticmethod
    def tanh_sinh(f, a, b, n=None, h=0.05):
        """
        Tanh-sinh quadrature — excellent near endpoint singularities.
        Transforms [a,b] via x = tanh(π/2 · sinh(t)).
        """
        # How many levels until weight < machine epsilon
        t_max = 3.5
        ts = np.arange(-t_max, t_max + h, h)
        sinh_t = np.sinh(ts)
        phi    = np.tanh(0.5*np.pi*sinh_t)
        dphi   = 0.5*np.pi*np.cosh(ts) / np.cosh(0.5*np.pi*sinh_t)**2
 
        # Map from (-1,1) to (a,b)
        x_ab = 0.5*(b - a)*phi + 0.5*(b + a)
        weights = dphi * 0.5*(b - a)
 
        # Mask out near-boundary where f might blow up
        mask = (x_ab > a + 1e-14) & (x_ab < b - 1e-14)
        x_ab = x_ab[mask]
        weights = weights[mask]
 
        y = np.array([f(xi) for xi in x_ab], dtype=float)
        # Replace NaN/Inf (singularities)
        bad = ~np.isfinite(y)
        y[bad] = 0.0
        return float(h * np.dot(weights, y))
