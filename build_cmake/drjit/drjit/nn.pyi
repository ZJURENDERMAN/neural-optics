from collections.abc import Iterator, Sequence as _Sequence
from typing import (
    Generic,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    Unpack,
    overload
)

import hashgrid

import drjit
import drjit.hashgrid as hashgrid
from drjit.hashgrid import (
    HashGridEncoding as HashGridEncoding,
    PermutoEncoding as PermutoEncoding
)
import drjit.random


class MatrixView:
    """
    The :py:class:`drjit.nn.MatrixView` provides pointer into a buffer along with
    shape and type metadata.

    Dr.Jit uses views to tightly pack sequences of matrices and bias vectors
    into a joint buffer, and to preserve information about the underlying data
    type and layout. The :py:func:`__getitem__` function can be used to slice a
    view into smaller sub-blocks.

    The typical process is to pack a PyTree of weight and bias vectors via
    :py:func:`drjit.pack()` into an inference or training-optimal
    representation. The returned views can then be passed to
    :py:func:`drjit.nn.matvec()`.
    """

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...

    def __getitem__(self, arg: Union[int, slice, Tuple[Union[int, slice], Union[int, slice]]]) -> MatrixView: ...

    @property
    def dtype(self) -> drjit.VarType:
        """Scalar type underlying the view."""

    @dtype.setter
    def dtype(self, arg: drjit.VarType, /) -> None: ...

    @property
    def offset(self) -> int:
        """Offset of the matrix data within ``buffer`` (counted in # of elements)"""

    @offset.setter
    def offset(self, arg: int, /) -> None: ...

    @property
    def stride(self) -> int:
        """Row stride (in # of elements)"""

    @stride.setter
    def stride(self, arg: int, /) -> None: ...

    @property
    def size(self) -> int:
        """Total number of elements"""

    @size.setter
    def size(self, arg: int, /) -> None: ...

    @property
    def layout(self) -> Literal['inference', 'training', 'row_major']:
        """
        One of several possible matrix layouts (training/inference-optimal and
        row-major).
        """

    @layout.setter
    def layout(self, value: Literal['inference', 'training', 'row_major']) -> None: ...

    @property
    def transpose(self) -> bool:
        """
        The ``MatrixView.T`` property flips this flag (all other
        values stay unchanged).
        """

    @transpose.setter
    def transpose(self, arg: bool, /) -> None: ...

    @property
    def shape(self) -> tuple[int, int]:
        """
        Number of rows/columns. Vectors are stored as matrices with one column.
        """

    @shape.setter
    def shape(self, arg: tuple[int, int], /) -> None: ...

    def __matmul__(self, arg: CoopVec[T], /) -> CoopVec[T]: ...

    @property
    def buffer(self) -> object:
        """
        The underlying buffer, which may contain additional matrices/vectors
        besides the data referenced by the :py:class:`MatrixView`.
        """

    @buffer.setter
    def buffer(self, arg: object, /) -> None: ...

    @property
    def T(self) -> MatrixView: ...

    @property
    def grad(self) -> MatrixView: ...

    DRJIT_STRUCT: dict = ...

class CoopVec(Generic[T]):
    def __init__(self, *args: Unpack[Tuple[Union[drjit.ArrayBase[SelfT, SelfCpT, ValT, ValCpT, T, PlainT, MaskT], float, int], ...]]) -> None:
        """
        The constructor accepts a variable number of arguments including Dr.Jit
        arrays, scalar Python integers and floating point values, and :ref:`PyTrees
        <pytrees>`. It flattens this input into a list of vector components.

        At least one Jit-compiled array must be provided as input so that Dr.Jit can
        infer the cooperative vector's element type. An exception will be raised if
        the input contains Dr.Jit arrays of inconsistent scalar types (e.g.,
        :py:class:`drjit.cuda.Array2f` and :py:class:`drjit.cuda.UInt`).
        """

    def __iter__(self, /) -> Iterator[T]: ...

    def __add__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]: ...

    def __radd__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]: ...

    def __sub__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]: ...

    def __rsub__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]: ...

    def __mul__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]: ...

    def __rmul__(self, arg: CoopVec[T] | T | float | int, /) -> CoopVec[T]: ...

    @property
    def index(self) -> int: ...

    @property
    def type(self) -> object: ...

    def __len__(self) -> int: ...

    def __abs__(self) -> CoopVec: ...

    def __repr__(self) -> str: ...

@overload
def pack(arg: MatrixView | drjit.AnyArray, *, layout: Literal['inference', 'training'] = 'inference') -> Tuple[drjit.ArrayBase, MatrixView]:
    """
    A training-optimal layout must be used used if the program *backpropagates*
    (as in :py:func:`dr.backward*() <drjit.backward>`) gradients through
    matrix-vector products. Inference (primal evaluation) and forward derivative
    propagation (as in :py:func:`dr.forward*() <drjit.forward>`) does not
    require a training-optimal layout.

    If the input matrices are already packed in a row-major layout, call
    :py:func:`dr.nn.view() <drjit.nn.view>` to create an efficient reference
    and then pass slices of the view to :py:func:`dr.nn.pack()
    <drjit.nn.pack>`. This avoids additional copies.

    .. code-block::

       mat: TensorXf = ...
       mat_view = dr.nn.view(mat)

       A1_view, A2_view = dr.nn.pack(
           mat_view[0:32, :],
           mat_view[32:64, :]
       )
    """

@overload
def pack(*args: PyTree, layout: Literal['inference', 'training'] = 'inference') -> Tuple[drjit.ArrayBase, Unpack[Tuple[PyTree, ...]]]: ...

@overload
def unpack(arg: MatrixView | drjit.AnyArray, /) -> Tuple[drjit.ArrayBase, MatrixView]:
    """
    The function :py:func:`dr.nn.unpack() <drjit.nn.unpack>` transforms a
    sequence (or :ref:`PyTree <pytrees>`) of vectors and optimal-layout matrices
    back into row-major layout.

    .. code-block:: python

       A_out, b_out = dr.nn.unpack(A_opt, b_opt)

    Note that the output of this function are (row-major) *views* into a shared
    buffer. Each view holds a reference to the shared buffer. Views can be
    converted back into regular tensors:

    .. code-block:: python

       A = TensorXf16(A)
    """

@overload
def unpack(*args: PyTree) -> Tuple[drjit.ArrayBase, Unpack[Tuple[PyTree, ...]]]: ...

def matvec(A: MatrixView, x: CoopVec[T], b: Optional[MatrixView] = None, /, transpose: bool = False) -> CoopVec[T]:
    """
    Evaluate a matrix-vector multiplication involving a cooperative vector.

    This function takes a *matrix view* ``A`` (see :py:func:`drjit.nn.pack`
    and :py:func:`drjit.nn.view` for details on views) and a *cooperative
    vector* ``x``. It then computes the associated matrix-vector product and
    returns it in the form of a new cooperative vector (potentially with a
    different size).

    The function can optionally apply an additive bias (i.e., to evaluate ``A@x
    + b``). This bias vector ``b`` should also be specified as a view.

    Specify ``tranpose=True`` to multiply by the transpose of the matrix ``A``.
    On the CUDA/OptiX backend, this feature requires that ``A`` is in inference
    or training-optimal layout.
    """

def view(arg: drjit.ArrayBase, /) -> MatrixView:
    """
    Convert a Dr.Jit array or tensor into a *view*.

    This function simply returns a view of the original tensor without
    transforming the underlying representation. This is useful to

    - Use :py:func:`drjit.nn.matvec` with a row-major matrix layout (which,
      however, is not recommended, since this can be significantly slower
      compared to matrices in inference/training-optimal layouts).

    - Slice a larger matrix into sub-blocks before passing them to
      :py:func:`drjit.nn.pack` (which also accepts *views* as inputs).
      This is useful when several matrices are already packed into a single
      matrix (which is, however, still in row-major layout). They can then be
      directly re-packed into optimal layouts without performing further
      unnecessary copies.
    """

def cast(arg0: CoopVec[T], arg1: Type[ArrayT], /) -> CoopVec[ArrayT]:
    """Cast the numeric type underlying a cooperative vector"""

T = TypeVar("T")

TensorOrViewOrNone: TypeAlias = Union[drjit.ArrayBase, MatrixView, None]

class Module:
    """
    This is the base class of a modular set of operations that make
    the specification of neural network architectures more convenient.

    Module subclasses are :ref:`PyTrees <pytrees>`, which means that various
    Dr.Jit operations can automatically traverse them.

    Constructing a neural network generally involves the following pattern:

    .. code-block::

       # 1. Establish the network structure
       net = nn.Sequential(
           nn.Linear(-1, 32, bias=False),
           nn.ReLU(),
           nn.Linear(-1, 3)
       )

       # 2. Instantiate the network for a specific backend + input size
       net = net.alloc(TensorXf16, 2)

       # 3. Pack coefficients into a training-optimal layout
       coeffs, net = nn.pack(net, layout='training')

    Network evaluation expects a :ref:`cooperative vector <coop_vec>` as input
    (i.e., ``net(nn.CoopVec(...))``) and returns another cooperative vector.
    The ``coeffs`` buffer contains all weight/bias data in training-optimal
    format and can be optimized, which will directly impact the packed network
    ``net`` that references this buffer.
    """

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        """
        Evaluate the model with an input cooperative vector and return the result.
        """

    def alloc(self, dtype: Type[drjit.ArrayBase], size: int = -1, rng: Optional[drjit.random.Generator] = None) -> Module:
        """
        Returns a new instance of the model with allocated weights.

        This function expects a suitable tensor ``dtype`` (e.g.
        :py:class:`drjit.cuda.ad.TensorXf16` or
        :py:class:`drjit.llvm.ad.TensorXf`) that will be used to store the
        weights on the device.

        If the model or one of its sub-models is automatically sized (e.g.,
        ``input_features=-1`` in :py:class:`drjit.nn.Linear`), the final
        network configuration may ambiguous and an exception will be raised.
        Specify the optional ``size`` parameter in such cases to inform the
        allocation about the size of the input cooperative vector.

        Layer weights are initialized using pseudorandom values obtained from
        the specified generator object ``rng``.

        Specifying a newly seeded random number generator with the same seed
        ensures that weights will be consistent across runs (i.e., calling
        ``alloc()`` twice will produce the same initialization).

        If ``rng=None`` (the default), a generator is constructed on the fly
        via ``dr.rng(seed=0x100000000)``. This particular seed value is used to
        de-correlate the network weights with respect to any potential future
        network evaluations that might be produced by a random number generator
        with the default seed (``0``). (Please ignore this paragraph if it
        is unclear, it explains a protection against a subtle/niche issue.)
        """

    def __repr__(self) -> str: ...

class Sequential(Module, _Sequence[Module]):
    """
    This model evaluates provided arguments ``arg[0]``, ``arg[1]``, ..., in sequence.
    """

    DRJIT_STRUCT: dict = {'layers' : tuple}

    def __init__(self, *args: Module): ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

    def __len__(self):
        """Return the number of contained models"""

    def __getitem__(self, index: int, /) -> Module:
        """Return the model at position ``index``"""

    def __repr__(self) -> str: ...

    __parameters__: tuple = ()

    __abstractmethods__: frozenset = ...

class ReLU(Module):
    r"""
    ReLU (rectified linear unit) activation function.

    This model evaluates the following expression:

    .. math::

       \mathrm{ReLU}(x) = \mathrm{max}\{x, 0\}.
    """

    DRJIT_STRUCT: dict = {}

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class LeakyReLU(Module):
    r"""
    "Leaky" ReLU (rectified linear unit) activation function.

    This model evaluates the following expression:

    .. math::

       \mathrm{LeakyReLU}(x) = \begin{cases}
          x,&\mathrm{if}\ x\ge 0,\\
          \texttt{negative\_slope}\cdot x,&\mathrm{otherwise}.
       \end{cases}
    """

    DRJIT_STRUCT: dict = ...

    def __init__(self, negative_slope: Union[float, drjit.ArrayBase] = 0.01): ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class Exp2(Module):
    r"""
    Applies the base-2 exponential function to each component.

    .. math::

       \mathrm{Exp2}(x) = 2^x

    On the CUDA backend, this function directly maps to an efficient native GPU instruction.
    """

    DRJIT_STRUCT: dict = {}

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class Exp(Module):
    r"""
    Applies the exponential function to each component.

    .. math::

       \mathrm{Exp}(x) = e^x
    """

    DRJIT_STRUCT: dict = {}

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class Tanh(Module):
    r"""
    Applies the hyperbolic tangent function to each component.

    .. math::

       \mathrm{Tanh}(x) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}

    On the CUDA backend, this function directly maps to an efficient native GPU instruction.
    """

    DRJIT_STRUCT: dict = {}

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class ScaleAdd(Module):
    r"""
    Scale the input by a fixed scale and apply an offset.

    Note that ``scale`` and ``offset`` are assumed to be constant (i.e., not trainable).

    .. math::

       \mathrm{ScaleAdd}(x) = x\cdot\texttt{scale} + \texttt{offset}
    """

    DRJIT_STRUCT: dict = ...

    def __init__(self, scale: Union[float, int, drjit.ArrayBase, None] = None, offset: Union[float, int, drjit.ArrayBase, None] = None): ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class Cast(Module):
    """
    Cast the input cooperative vector to a different precision. Should be
    instantiated with the desired element type, e.g. ``Cast(drjit.cuda.ad.Float32)``
    """

    DRJIT_STRUCT: dict = {'dtype' : Union[type[drjit.ArrayBase], None]}

    def __init__(self, dtype: Optional[Type[drjit.ArrayBase]] = None): ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

    def __repr__(self): ...

class Linear(Module):
    r"""
    This layer represents a learnable affine linear transformation of the input
    data following the expression :math:`\mathbf{y} = \mathbf{A}\mathbf{x} +
    \mathbf{b}`.

    It takes ``in_features`` inputs and returns a cooperative vector with
    ``out_features`` dimensions. The following parameter values have a special
    a meaning:

    - ``in_features=-1``: set the input size to match the previous model's
      output (or the input of the network, if there is no previous model).

    - ``out_features=-1``: set the output size to match the input size.

    The bias (:math:`\textbf{b}`) term is optional and can be disabled by
    specifying ``bias=False``.

    The method :py:func:`Module.alloc` initializes the underlying coefficient
    storage with random weights following a uniform Xavier initialization,
    i.e., uniform variates on the interval :math:`[-k,k]` where
    :math:`k=1/\sqrt{\texttt{out\_features}}`.
    """

    DRJIT_STRUCT: dict = ...

    def __init__(self, in_features: int = -1, out_features: int = -1, bias=True) -> None: ...

    def __repr__(self) -> str: ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class TriEncode(Module):
    r"""
    Map an input onto a higher-dimensional space by transforming it using
    triangular sine and cosine approximations of an increasing frequency.

    .. math::

       x\mapsto \begin{bmatrix}
           \sin_\triangle(2^0\,x)\\
           \cos_\triangle(2^0\,x)\\
           \vdots\\
           \cos_\triangle(2^{n-1}\, x)\\
           \sin_\triangle(2^{n-1}\, x)
       \end{bmatrix}

    where

    .. math::

       \cos_\triangle(x) = 1-4\left|x-\mathrm{round}(x)\right|

    and

    .. math::

       \sin_\triangle(x) = \cos_\triangle(x-1/4)

    The value :math:`n` refers to the number of *octaves*. This layer increases
    the dimension by a factor of :math:`2n`.

    Note that this encoding has period 1. If your input exceeds the interval
    :math:`[0, 1]`, it is advisable that you reduce it to this range to avoid
    losing information.

    Minima/maxima of higher frequency components conincide on a regular
    lattice, which can lead to reduced fitting performance at those locations.
    Specify the optional parameter ``shift`` to phase-shift the :math:`i`-th
    frequency by :math:`2\,\pi\,\mathrm{shift}` to avoid this behavior.

    The following plot shows the first two octaves applied to the linear
    function on :math:`[0, 1]` (without shift).

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/tri_encode_light.svg
      :class: only-light
      :width: 600px
      :align: center

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/tri_encode_dark.svg
      :class: only-dark
      :width: 600px
      :align: center
    """

    DRJIT_STRUCT: dict = ...

    def __init__(self, octaves: int = 0, shift: float = 0) -> None: ...

    def __repr__(self) -> str: ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class SinEncode(Module):
    r"""
    Map an input onto a higher-dimensional space by transforming it using sines
    and cosines of an increasing frequency.

    .. math::

       x\mapsto \begin{bmatrix}
           \sin(2^0\, 2\pi x)\\
           \cos(2^0\, 2\pi x)\\
           \vdots\\
           \sin(2^{n-1}\, 2\pi x)\\
           \cos(2^{n-1}\, 2\pi x)\\
       \end{bmatrix}


    The value :math:`n` refers to the number of *octaves*. This layer increases
    the dimension by a factor of :math:`2n`.

    Note that this encoding has period 1. If your input exceeds the interval
    :math:`[0, 1]`, it is advisable that you reduce it to this range to avoid
    losing information.

    Minima/maxima of higher frequency components conincide on a regular
    lattice, which can lead to reduced fitting performance at those locations.
    Specify the optional parameter ``shift`` to phase-shift the :math:`i`-th
    frequency by :math:`\mathrm{shift}` radians to avoid this behavior.

    The following plot shows the first two octaves applied to the linear
    function on :math:`[0, 1]` (without shift).

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/sin_encode_light.svg
      :class: only-light
      :width: 600px
      :align: center

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/sin_encode_dark.svg
      :class: only-dark
      :width: 600px
      :align: center
    """

    DRJIT_STRUCT: dict = ...

    def __init__(self, octaves: int = 0, shift: float = 0) -> None: ...

    def __repr__(self) -> str: ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

class HashEncodingLayer(Module):
    """
    Simple layer wrapping a hash encoding like :py:class:`drjit.nn.HashGridEncoding`
    or :py:class:`drjit.nn.PermutoEncoding`.

    Note that the parameters of the encoding will not be included when packing the
    network, as the data representations are generally incompatible. You must initialize
    the encoding parameters separately.
    """

    def __init__(self, encoding: hashgrid.HashEncoding) -> None: ...

    def __call__(self, arg: CoopVec, /) -> CoopVec: ...

    @property
    def data(self): ...

    def __repr__(self) -> str: ...
