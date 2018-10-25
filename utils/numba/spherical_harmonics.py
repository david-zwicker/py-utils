
import math
from cmath import sqrt as csqrt

from scipy.special import sph_harm
from numba import jit



def spherical_index_k(l, m=0):
    """ returns the mode k from the degree l and order m """
    if not -l <= m <= l:
        raise ValueError('m must lie between -l and l')
    return l*(l + 1) + m



def spherical_index_lm(k):
    """ returns the degree l and the order m from the mode k """
    l = int(math.floor(math.sqrt(k)))
    return l, k - l*(l + 1)



def spherical_harmonic_symmetric_scipy(l, theta):
    """ axisymmetric spherical harmonics with degree l"""
    return (sph_harm(0., l, 0., theta)).real



def spherical_harmonic_real_scipy(l, m, theta, phi):
    """ real spherical harmonics of degree l and order m """
    # note that the definition of `sph_harm` has a different convention for the
    # usage of the variables phi and theta
    if m > 0:
        return (sph_harm(m, l, phi, theta) +
                (-1)**m * sph_harm(-m, l, phi, theta)
                ).real / math.sqrt(2)
        
    elif m == 0:
        return sph_harm(0, l, phi, theta).real
    
    else:  # m < 0
        return ((sph_harm(-m, l, phi, theta) -
                 (-1)**m * sph_harm(m, l, phi, theta)
                 ) / csqrt(-2)).real



def spherical_harmonic_real_scipy_k(k, theta, phi):
    """ real spherical harmonics described by mode k """
    return spherical_harmonic_real_scipy(*spherical_index_lm(k), theta=theta,
                                         phi=phi)



# =============================================================================
# Automatically generated code from docs/mathematica/SphericalHarmonics.nb
# =============================================================================



MAX_ORDER = 5  # maximally supported order
MAX_ORDER_SYM = 7  # maximally supported order for symmetric case
NotANumber = float('nan')



@jit("f8(i8, f8)")
def spherical_harmonic_symmetric(n, theta):
    """ Axisymmetric spherical harmonics of degree n """

    x = math.cos(theta)
    if n == 0:
        return 0.28209479177387814
    elif n == 1:
        return 0.4886025119029199*x
    elif n == 2:
        return -0.31539156525252 + 0.9461746957575601*x*x
    elif n == 3:
        return x*(-1.1195289977703462 + 1.865881662950577*x*x)
    elif n == 4:
        return 0.31735664074561293 + x*x*(-3.173566407456129 + 3.7024941420321507*x*x)
    elif n == 5:
        return x*(1.754254836801354 + x*x*(-8.186522571739653 + 7.367870314565687*x*x))
    elif n == 6:
        return -0.3178460113381421 + x*x*(6.674766238100984 + x*x*(-20.02429871430295 + 14.684485723822165*x*x))
    elif n == 7:
        return x*(-2.3899496919201733 + x*x*(21.50954722728156 + x*x*(-47.32100390001943 + 29.293954795250123*x*x)))
    else:
        return NotANumber



@jit("f8(i8, i8, f8, f8)")
def spherical_harmonic_real(l, m, theta, phi):
    """ Real spherical harmonics of degree l and order m """
    if l == 0:
        if m == 0:
            return 0.28209479177387814
        else:
            return 0
    elif l == 1:
        if m == -1:
            st = math.sin(theta)
            return -0.4886025119029199 * st * math.sin(phi)
        elif m == 0:
            ct = math.cos(theta)
            return 0.4886025119029199 * ct
        elif m == 1:
            st = math.sin(theta)
            return -0.4886025119029199 * st * math.cos(phi)
        else:
            return 0
    elif l == 2:
        if m == -2:
            st = math.sin(theta)
            return 0.5462742152960396 * st**2 * math.sin(2. * phi)
        elif m == -1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return -1.0925484305920792 * ct * st * math.sin(phi)
        elif m == 0:
            ct = math.cos(theta)
            return 0.31539156525252005 * (-1. + 3. * ct**2)
        elif m == 1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return -1.0925484305920792 * ct * st * math.cos(phi)
        elif m == 2:
            st = math.sin(theta)
            return 0.5462742152960396 * st**2 * math.cos(2. * phi)
        else:
            return 0
    elif l == 3:
        if m == -3:
            st = math.sin(theta)
            return -0.5900435899266435 * st**3 * math.sin(3. * phi)
        elif m == -2:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 1.445305721320277 * ct * st**2 * math.sin(2. * phi)
        elif m == -1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.4570457994644658 * (1. - 5. * ct**2) * st * math.sin(phi)
        elif m == 0:
            ct = math.cos(theta)
            return 0.3731763325901154 * ct * (-3. + 5. * ct**2)
        elif m == 1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.4570457994644658 * (1. - 5. * ct**2) * st * math.cos(phi)
        elif m == 2:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 1.445305721320277 * ct * st**2 * math.cos(2. * phi)
        elif m == 3:
            st = math.sin(theta)
            return -0.5900435899266435 * st**3 * math.cos(3. * phi)
        else:
            return 0
    elif l == 4:
        if m == -4:
            st = math.sin(theta)
            return 0.6258357354491761 * st**4 * math.sin(4. * phi)
        elif m == -3:
            st = math.sin(theta)
            ct = math.cos(theta)
            return -1.7701307697799304 * ct * st**3 * math.sin(3. * phi)
        elif m == -2:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.47308734787878004 * (-1. + 7. * ct**2) * st**2 * math.sin(2. * phi)
        elif m == -1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.6690465435572892 * ct * (3. - 7. * ct**2) * st * math.sin(phi)
        elif m == 0:
            ct = math.cos(theta)
            return 0.10578554691520431 * (3. - 30. * ct**2 + 35. * ct**4)
        elif m == 1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.6690465435572892 * ct * (3. - 7. * ct**2) * st * math.cos(phi)
        elif m == 2:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.47308734787878004 * (-1. + 7. * ct**2) * st**2 * math.cos(2. * phi)
        elif m == 3:
            st = math.sin(theta)
            ct = math.cos(theta)
            return -1.7701307697799304 * ct * st**3 * math.cos(3. * phi)
        elif m == 4:
            st = math.sin(theta)
            return 0.6258357354491761 * st**4 * math.cos(4. * phi)
        else:
            return 0
    elif l == 5:
        if m == -5:
            st = math.sin(theta)
            return -0.6563820568401701 * st**5 * math.sin(5. * phi)
        elif m == -4:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 2.0756623148810416 * ct * st**4 * math.sin(4. * phi)
        elif m == -3:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.4892382994352504 * (1. - 9. * ct**2) * st**3 * math.sin(3. * phi)
        elif m == -2:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 2.396768392486662 * ct * (-1. + 3. * ct**2) * st**2 * math.sin(2. * phi)
        elif m == -1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return -0.45294665119569694 * (1. - 14. * ct**2 + 21. * ct**4) * st * math.sin(phi)
        elif m == 0:
            ct = math.cos(theta)
            return 0.1169503224534236 * ct * (15. - 70. * ct**2 + 63. * ct**4)
        elif m == 1:
            st = math.sin(theta)
            ct = math.cos(theta)
            return -0.45294665119569694 * (1. - 14. * ct**2 + 21. * ct**4) * st * math.cos(phi)
        elif m == 2:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 2.396768392486662 * ct * (-1. + 3. * ct**2) * st**2 * math.cos(2. * phi)
        elif m == 3:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 0.4892382994352504 * (1. - 9. * ct**2) * st**3 * math.cos(3. * phi)
        elif m == 4:
            st = math.sin(theta)
            ct = math.cos(theta)
            return 2.0756623148810416 * ct * st**4 * math.cos(4. * phi)
        elif m == 5:
            st = math.sin(theta)
            return -0.6563820568401701 * st**5 * math.cos(5. * phi)
        else:
            return 0
    else:
        return NotANumber



@jit("f8(i8, f8, f8)")
def spherical_harmonic_real_k(k, theta, phi):
    """ Real spherical harmonics of degree l and order m, described by a index k = m + l*(l + 1) """
    if k == 0:
        return 0.28209479177387814
    elif k == 1:
        st = math.sin(theta)
        return -0.4886025119029199 * st * math.sin(phi)
    elif k == 2:
        ct = math.cos(theta)
        return 0.4886025119029199 * ct
    elif k == 3:
        st = math.sin(theta)
        return -0.4886025119029199 * st * math.cos(phi)
    elif k == 4:
        st = math.sin(theta)
        return 0.5462742152960396 * st**2 * math.sin(2. * phi)
    elif k == 5:
        st = math.sin(theta)
        ct = math.cos(theta)
        return -1.0925484305920792 * ct * st * math.sin(phi)
    elif k == 6:
        ct = math.cos(theta)
        return 0.31539156525252005 * (-1. + 3. * ct**2)
    elif k == 7:
        st = math.sin(theta)
        ct = math.cos(theta)
        return -1.0925484305920792 * ct * st * math.cos(phi)
    elif k == 8:
        st = math.sin(theta)
        return 0.5462742152960396 * st**2 * math.cos(2. * phi)
    elif k == 9:
        st = math.sin(theta)
        return -0.5900435899266435 * st**3 * math.sin(3. * phi)
    elif k == 10:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 1.445305721320277 * ct * st**2 * math.sin(2. * phi)
    elif k == 11:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.4570457994644658 * (1. - 5. * ct**2) * st * math.sin(phi)
    elif k == 12:
        ct = math.cos(theta)
        return 0.3731763325901154 * ct * (-3. + 5. * ct**2)
    elif k == 13:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.4570457994644658 * (1. - 5. * ct**2) * st * math.cos(phi)
    elif k == 14:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 1.445305721320277 * ct * st**2 * math.cos(2. * phi)
    elif k == 15:
        st = math.sin(theta)
        return -0.5900435899266435 * st**3 * math.cos(3. * phi)
    elif k == 16:
        st = math.sin(theta)
        return 0.6258357354491761 * st**4 * math.sin(4. * phi)
    elif k == 17:
        st = math.sin(theta)
        ct = math.cos(theta)
        return -1.7701307697799304 * ct * st**3 * math.sin(3. * phi)
    elif k == 18:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.47308734787878004 * (-1. + 7. * ct**2) * st**2 * math.sin(2. * phi)
    elif k == 19:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.6690465435572892 * ct * (3. - 7. * ct**2) * st * math.sin(phi)
    elif k == 20:
        ct = math.cos(theta)
        return 0.10578554691520431 * (3. - 30. * ct**2 + 35. * ct**4)
    elif k == 21:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.6690465435572892 * ct * (3. - 7. * ct**2) * st * math.cos(phi)
    elif k == 22:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.47308734787878004 * (-1. + 7. * ct**2) * st**2 * math.cos(2. * phi)
    elif k == 23:
        st = math.sin(theta)
        ct = math.cos(theta)
        return -1.7701307697799304 * ct * st**3 * math.cos(3. * phi)
    elif k == 24:
        st = math.sin(theta)
        return 0.6258357354491761 * st**4 * math.cos(4. * phi)
    elif k == 25:
        st = math.sin(theta)
        return -0.6563820568401701 * st**5 * math.sin(5. * phi)
    elif k == 26:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 2.0756623148810416 * ct * st**4 * math.sin(4. * phi)
    elif k == 27:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.4892382994352504 * (1. - 9. * ct**2) * st**3 * math.sin(3. * phi)
    elif k == 28:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 2.396768392486662 * ct * (-1. + 3. * ct**2) * st**2 * math.sin(2. * phi)
    elif k == 29:
        st = math.sin(theta)
        ct = math.cos(theta)
        return -0.45294665119569694 * (1. - 14. * ct**2 + 21. * ct**4) * st * math.sin(phi)
    elif k == 30:
        ct = math.cos(theta)
        return 0.1169503224534236 * ct * (15. - 70. * ct**2 + 63. * ct**4)
    elif k == 31:
        st = math.sin(theta)
        ct = math.cos(theta)
        return -0.45294665119569694 * (1. - 14. * ct**2 + 21. * ct**4) * st * math.cos(phi)
    elif k == 32:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 2.396768392486662 * ct * (-1. + 3. * ct**2) * st**2 * math.cos(2. * phi)
    elif k == 33:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 0.4892382994352504 * (1. - 9. * ct**2) * st**3 * math.cos(3. * phi)
    elif k == 34:
        st = math.sin(theta)
        ct = math.cos(theta)
        return 2.0756623148810416 * ct * st**4 * math.cos(4. * phi)
    elif k == 35:
        st = math.sin(theta)
        return -0.6563820568401701 * st**5 * math.cos(5. * phi)
    else:
        return NotANumber
    