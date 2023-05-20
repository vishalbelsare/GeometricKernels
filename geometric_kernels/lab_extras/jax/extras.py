from typing import List

import jax.numpy as jnp
import lab as B
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.JAXNumeric]


@dispatch
def take_along_axis(a: Union[_Numeric, B.Numeric], index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    if not isinstance(a, jnp.ndarray):
        a = jnp.array(a)
    return jnp.take_along_axis(a, index, axis=axis)


@dispatch
def from_numpy(_: B.JAXNumeric, b: Union[List, B.NPNumeric, B.Number, B.JAXNumeric]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return jnp.array(b)


@dispatch
def trapz(y: B.JAXNumeric, x: _Numeric, dx: _Numeric = 1.0, axis: int = -1):  # type: ignore
    """
    Integrate along the given axis using the trapezoidal rule.
    """
    return jnp.trapz(y, x, dx, axis)


@dispatch
def logspace(start: B.JAXNumeric, stop: _Numeric, num: int = 50):  # type: ignore
    """
    Return numbers spaced evenly on a log scale.
    """
    return jnp.logspace(start, stop, num)


@dispatch
def degree(a: B.JAXNumeric):  # type: ignore
    """
    Given an adjacency matrix `a`, return a diagonal matrix
    with the col-sums of `a` as main diagonal - this is the
    degree matrix representing the number of nodes each node
    is connected to.
    """
    degrees = a.sum(axis=0)  # type: ignore
    return jnp.diag(degrees)


@dispatch
def eigenpairs(L: B.JAXNumeric, k: int):
    """
    Obtain the k highest eigenpairs of a symmetric PSD matrix L.
    """
    l, u = jnp.linalg.eigh(L)
    return l[:k], u[:, :k]


@dispatch
def set_value(a: B.JAXNumeric, index: int, value: float):
    """
    Set a[index] = value.
    This operation is not done in place and a new array is returned.
    """
    a = a.at[index].set(value)
    return a


@dispatch
def dtype_double(reference: B.JAXRandomState):  # type: ignore
    """
    Return `double` dtype of a backend based on the reference.
    """
    return jnp.float64


@dispatch
def dtype_integer(reference: B.JAXRandomState):  # type: ignore
    """
    Return `int` dtype of a backend based on the reference.
    """
    return jnp.int32


@dispatch
def get_random_state(key: B.JAXRandomState):
    """
    Return the random state of a random generator.

    Parameters
    ----------
    key : B.JAXRandomState
        The key used to generate the random state.

    Returns
    -------
    Any
        The random state of the random generator.
    """
    return key


@dispatch
def set_random_state(key: B.JAXRandomState, state):
    """
    Set the random state of a random generator.

    Parameters
    ----------
    key : B.JAXRandomState
        The random generator.
    state : Any
        The new random state of the random generator.
    """
    pass


@dispatch
def create_complex(real: _Numeric, imag: B.JAXNumeric):
    """
    Returns a complex number with the given real and imaginary parts using jax.

    Args:
    - real: float, real part of the complex number.
    - imag: float, imaginary part of the complex number.

    Returns:
    - complex_num: complex, a complex number with the given real and imaginary parts.
    """
    complex_num = real + 1j * imag
    return complex_num


@dispatch
def dtype_complex(reference: B.JAXNumeric):
    """
    Return `complex` dtype of a backend based on the reference.
    """
    if B.dtype(reference) == jnp.float32:
        return jnp.complex64
    else:
        return jnp.complex128


@dispatch
def cumsum(x: B.JAXNumeric, axis=None):
    """
    Return cumulative sum (optionally along axis)
    """
    return jnp.cumsum(x, axis=axis)


@dispatch
def qr(x: B.JAXNumeric):
    """
    Return a QR decomposition of a matrix x.
    """
    Q, R = jnp.linalg.qr(x)
    return Q, R


@dispatch
def slogdet(x: B.JAXNumeric):
    """
    Return the sign and log-determinant of a matrix x.
    """
    sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


@dispatch
def eigvalsh(x: B.JAXNumeric):
    """
    Compute the eigenvalues of a Hermitian or real symmetric matrix x.
    """
    return jnp.linalg.eigvalsh(x)
