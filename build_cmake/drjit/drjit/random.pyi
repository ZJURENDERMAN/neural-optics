from typing import TypeAlias, TypeVar, Union

import drjit as dr
import drjit


ArrayT = TypeVar("ArrayT", bound=drjit.ArrayBase)

Shape: TypeAlias = Union[int, tuple[int, Ellipsis]]

ArrayOrInt: TypeAlias = Union[int, drjit.ArrayBase]

ArrayOrFloat: TypeAlias = Union[float, drjit.ArrayBase]

class Generator:
    def random(self, dtype: type[ArrayT], shape: Union[int, tuple[int, Ellipsis]]) -> ArrayT:
        """
        Return a Dr.Jit array or tensor containing uniformly distributed
        pseudorandom variates.

        This function supports floating point arrays/tensors of various
        configurations and precisions, e.g.:

        .. code-block:: python

           from drjit.cuda import Float, TensorXf, Array3f, Matrix4f

           # Example usage
           rng = dr.rng(seed=0)
           rand_array = rng.random(Float, 128)
           rand_tensor = rng.random(TensorXf16, shape=(128, 128))
           rand_vec = rng.random(Array3f, (3, 128))
           rand_mat = rng.random(Matrix4f64, (4, 4, 128))

        The output is uniformly distributed the half-open interval :math:`[0, 1)`.
        Integer arrays are not supported.

        Args:
            source (type[ArrayT]): A Dr.Jit tensor or array type.

            shape (int | tuple[int, ...]): The target shape

        Returns:
            ArrayT: The generated array of random variates.
        """

    def uniform(self, dtype: type[ArrayT], shape: Union[int, tuple[int, Ellipsis]], low: Union[float, drjit.ArrayBase] = 0.0, high: Union[float, drjit.ArrayBase] = 1.0):
        """
        Return a Dr.Jit array or tensor containing uniformly distributed
        pseudorandom variates.

        This function resembles :py:func:`random()` but additionally ensures
        that variates are distributed on the half-open interval
        :math:`[        exttt{low},     exttt{high})`.

        Args:
            source (type[ArrayT]): A Dr.Jit tensor or array type.

            shape (int | tuple[int, ...]): The target shape

            low (float | drjit.ArrayBase): The low value of the desired interval

            high (float | drjit.ArrayBase): The high value of the desired interval

        Returns:
            ArrayT: The generated array of random variates.
        """

    def normal(self, dtype: type[ArrayT], shape: Union[int, tuple[int, Ellipsis]], loc: Union[float, drjit.ArrayBase] = 0.0, scale: Union[float, drjit.ArrayBase] = 1.0) -> ArrayT:
        """
        Return a Dr.Jit array or tensor containing pseudorandom variates
        following a standard normal distribution

        This function supports arrays/tensors of various configurations and
        precisions--see the similar :py:func:`drjit.random()` for examples on
        how to call this function.

        Args:
            source (type[ArrayT]): A Dr.Jit tensor or array type.

            shape (int | tuple[int, ...]): The target shape

            loc (float | drjit.ArrayBase): The mean of the normal distribution (``0.0`` by default)

            scale (float | drjit.ArrayBase): The standard deviation of the normal distribution (``1.0`` by default)

        Returns:
            ArrayT: The generated array of random variates.
        """

    def clone(self) -> Generator: ...

class Philox4x32Generator(Generator):
    """
    Implementation of the :py:class:`Generator` interface based on the Philox4x32 RNG.
    """

    DRJIT_STRUCT: dict = ...

    def __init__(self, seed: Union[int, drjit.ArrayBase] = 0, counter: Union[int, drjit.ArrayBase] = 0): ...

    def clone(self) -> Generator: ...

    def random(self, dtype: type[ArrayT], shape: Union[int, tuple[int, Ellipsis]]) -> ArrayT: ...

    def normal(self, dtype: type[ArrayT], shape: Union[int, tuple[int, Ellipsis]], loc: Union[float, drjit.ArrayBase] = 0.0, scale: Union[float, drjit.ArrayBase] = 1.0) -> ArrayT: ...

    def __repr__(self) -> str: ...
