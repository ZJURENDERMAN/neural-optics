import dr

import drjit as dr
import drjit
import drjit.random


def cosine_ramp(x: dr.ArrayBase) -> dr.ArrayBase:
    """"Smoothed" ramp to help features blend-in without instabilities"""

def div_round_up(num: int, divisor: int) -> int:
    """Compute ceiling division (num / divisor) using integer arithmetic."""

def next_multiple(num: int, multiple: int) -> int:
    """Round `num` to the next multiple of `multiple`"""

class HashEncoding:
    """
    This class serves as the interface for Hash based encodings. It is the base
    class for both the ``HashGridEncoding``, as well as the
    ``PermutoEncoding``, and stores various fields that are used by both of
    them.
    """

    DRJIT_STRUCT: dict = {'data' : drjit.ArrayBase}

    def __init__(self, dtype: Type[dr.ArrayBase], dimension: int, *, n_levels: int = 16, n_features_per_level: int = 2, hashmap_size: int = 524288, base_resolution: int = 16, per_level_scale: float = 2, align_corners: bool = False, torchngp_compat: bool = False, smooth_weight_gradients: bool = False, smooth_weight_lambda: float = 1.0, init_scale: float = 0.0001, rng: dr.random.Generator | None = None) -> None:
        """
        Initialize a hash encoding. This computes fields used by both HashGrid
        and Permutohedral encodings, as well as defining types used throughout
        the encodings.

        Args:
            dimension: The dimensionality of the hash encoding. This corresponds to
                the number of input features the encoding can take.
            n_levels: Hash encodings generally make use of multiple levels of the same
                encoding with different scales. This parameter specifies the number of
                levels used by this encoding.
            n_features_per_level: Specifies how many features are stored at each vertex
                and at each level. The number of output features of the hash encoding
                is given by ``n_levels * n_features_per_level``. In order to ensure efficient
                gradient backpropagation, this value should be a multiple of two.
            hashmap_size: Specifies the maximal number of parameters per level of the
                hash encoding. HashGrids will use a dense grid lookup for layers with
                a low enough scale, and use less than ``hashmap_size`` number of parameters
                per level.
            base_resolution: The scale factor of the 0th layer in the hash encoding.
            per_level_scale: To calculate the scale of a layer, the scale of the previous
                layer is multiplied by this value.
            align_corners: If this value is ``True``, the simplex vertices are aligned
                with the domain of the encoding [0, 1].
            smooth_weight_gradients: whether to smooth the gradients of the weights
                by using a straight-through estimator.
            smooth_weight_lambda: the value of lambda used for the straight-through estimator.
            init_scale: The parameters of the hashgrid are initialized with a uniform
                distribution, ranging from -init_scale to +init_scale.
            rng: Random number generator, used to initialize the parameters.
        """

    @property
    def n_params(self) -> int:
        """The number of parameters held by this encoding."""

    def set_params(self, values: dr.ArrayBase) -> None:
        """
        This function can be used to set the parameters of the hashgrid. It can
        be used to update parameters from the optimizer.
        """

    @property
    def params(self) -> dr.ArrayBase:
        """The parameters stored by this encoding."""

    @params.setter
    def params(self, values):
        """Setter for the parameters of this hashgrid."""

    @property
    def dtype(self) -> int:
        """The Dr.Jit type of the parameters used by this hashgrid."""

    @property
    def dimension(self) -> int:
        """The dimensionality of this hash encoding."""

    @property
    def hashmap_size(self) -> int:
        """The hashmap size provided when constructing this encoding."""

    @property
    def n_levels(self) -> int:
        """
        The number of levels in this hash encoding.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """

    @property
    def base_resolution(self) -> int:
        """The resolution of the 0th level."""

    @property
    def per_level_scale(self) -> float:
        """The per level scale factor, with which the scale of each level grows."""

    @property
    def n_features_per_level(self) -> int:
        """
        The number of features per level.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """

    @property
    def align_corners(self) -> bool:
        """
        If the corners of the hashgrid should be aligned to the edges of its domain.
        """

    @property
    def torchngp_compat(self) -> int:
        """
        Enable tiny-cuda-nn compatible indexing and offset calculations.

        When True, uses the same indexing functions, stride calculations, and
        position offsets as the reference tiny-cuda-nn implementation for
        reproducible results across implementations.
        """

    @property
    def smooth_weight_gradients(self) -> bool:
        """Whether to apply gradient smoothing to weights."""

    @property
    def smooth_weight_lambda(self) -> float:
        """
        Blending factor for gradient smoothing using straight-through estimator.

        Controls the strength of the gradient smoothing applied to interpolation weights.
        A value of 1.0 fully replaces the original gradients with smoothed ones,
        while 0.0 disables smoothing entirely.
        """

    @property
    def init_scale(self) -> float:
        """
        Scale for uniform random initialization of parameters.

        When allocating a hash encoding, the parameters are initialized using a
        uniform random distribution. This value is used to scale this distribution,
        ranging from -init_scale to +init_scale.
        """

    @property
    def n_output_features(self) -> int:
        """
        The total number of output features ``n_levels * n_features_per_level``
        for this encoding.

        This is computed as n_levels * n_features_per_level, representing
        the concatenation of features from all resolution levels. For example,
        with 16 levels and 2 features per level, this returns 32.
        """

    def indexing_function(self, key: dr.ArrayBase, level_i: int) -> dr.ArrayBase:
        """
        Given a key i.e. a D-dimensional integer vector identifying a vertex
        of a simplex, this function calculates the index of the feature tuple,
        which is associated with that vertex.
        """

    def hash(self, key) -> dr.ArrayBase:
        """
        Hashes the D-dimensional key to compute a 1-dimensional index. This function
        is called when dense indexing is not possible.
        """

class HashGridEncoding(HashEncoding):
    """
    This encoding implements a Multiresolution Hash Grid. For every resolution level,
    this encoding looks up the :math:`2^D` vertices of the cell in which the input point is
    located, performs multilinear interpolation, and concatenates the features accross
    all resolution levels.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: Specifies how many features are stored at each vertex
            and at each level. The number of output features of the hash encoding
            is given by ``n_levels * n_features_per_level``. In order to ensure efficient
            gradient backpropagation, this value should be a multiple of two.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
        init_scale: The parameters of the hashgrid are initialized with a uniform
            distribution, ranging from -init_scale to +init_scale.
        rng: Random number generator, used to initialize the parameters.
    """

    def __init__(self, dtype: Type[drjit.ArrayBase], dimension: int, *, n_levels: int = 16, n_features_per_level: int = 2, hashmap_size: int = 524288, base_resolution: int = 16, per_level_scale: float = 2, align_corners: bool = False, torchngp_compat: bool = False, smooth_weight_gradients: bool = False, smooth_weight_lambda: float = 1.0, init_scale: float = 0.0001, rng: drjit.random.Generator | None = None) -> None: ...

    def __call__(self, p: Iterable[drjit.ArrayBase], active: bool | drjit.ArrayBase = True) -> Iterable[drjit.ArrayBase]: ...

    def hash(self, key): ...

    def __repr__(self) -> str: ...

    def level_offset(self, level_i: int) -> int:
        """
        Helpful to build e.g. debug visualizations by splitting params per level.
        Must be called with an index up to and including `n_levels`.

        Warning: the level offset is expressed in number of vertices, i.e. it
        does *not* account for the feature count in each vertex. Each level contains
        `n_features_per_level` times the difference between the next offset entries.
        """

class PermutoEncoding(HashEncoding):
    """
    Permutohedral lattice-based encoding inspired by the paper `PermutoSDF Fast
    Multi-View Reconstruction with Implicit Surfaces using Permutohedral
    Lattices <https://radualexandru.github.io/permuto_sdf>`__.

    Unlike hash grid encodings that use regular grid lattices, this encoding employs
    a permutohedral lattice structure where simplices consist of triangles, tetrahedra,
    and higher-dimensional analogs. The key advantage is linear scaling: the number of
    vertices per simplex (and thus memory lookups per sample per level) grows linearly
    with dimensionality, compared to exponential growth in grid-based approaches.

    This implementation by `Tobias Zirr <https://github.com/tszirr>`__
    simplifies the original method by performing sorting and interpolation
    directly in :math:`d`-dimensional space, avoiding the elevation to a hyperplane in
    :math:`(d+1)`-dimensional space used in the reference implementation.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: Specifies how many features are stored at each vertex
            and at each level. The number of output features of the hash encoding
            is given by ``n_levels * n_features_per_level``. In order to ensure efficient
            gradient backpropagation, this value should be a multiple of two.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
        init_scale: The parameters of the hashgrid are initialized with a uniform
            distribution, ranging from -init_scale to +init_scale.
        rng: Random number generator, used to initialize the parameters.
    """

    def __init__(self, dtype: Type[drjit.ArrayBase], dimension: int, *, n_levels: int = 16, n_features_per_level: int = 2, hashmap_size: int = 524288, base_resolution: int = 16, per_level_scale: float = 2, align_corners: bool = False, smooth_weight_gradients: bool = False, smooth_weight_lambda: float = 1.0, init_scale: float = 0.0001, rng: dr.random.Generator | None = None) -> None: ...

    def __call__(self, p: Iterable[drjit.ArrayBase], active: bool | drjit.ArrayBase = True) -> Iterable[drjit.ArrayBase]: ...

    def hash(self, key) -> dr.ArrayBase:
        """
        Polynomial rolling hash for mapping lattice coordinates to hash table indices.

        Uses a simple multiplicative hash with prime number 2531011 to distribute
        lattice coordinates uniformly across the hash table space.
        """

    def __repr__(self) -> str: ...
