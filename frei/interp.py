import numpy as np
import xarray
import numba as nb
import pandas as pd
from numpy_groupies.aggregate_numba import (
    Sum, Prod, Len, All, Any, Last, First, AllNan, AnyNan, Min,
    Max, ArgMin, ArgMax, Mean, Std, Var,
    CumSum, CumProd, CumMax, CumMin, funcs_no_separate_nan,
    check_dtype, check_fill_value,
    get_func, isstr, _default_cache
)
from numpy_groupies.utils_numpy import input_validation, get_aliasing


__all__ = ['groupby_bins_agg']


class AggregateOp(object):
    """
    Every subclass of AggregateOp handles a different aggregation operation. There are
    several private class methods that need to be overwritten by the subclasses
    in order to implement different functionality.

    On object instantiation, all necessary static methods are compiled together into
    two jitted callables, one for scalar arguments, and one for arrays. Calling the
    instantiated object picks the right cached callable, does some further preprocessing
    and then executes the actual aggregation operation.
    """

    forced_fill_value = None
    counter_fill_value = 1
    counter_dtype = bool
    mean_fill_value = None
    mean_dtype = np.float64
    outer = False
    reverse = False
    nans = False

    def __init__(self, func=None, **kwargs):
        if func is None:
            func = type(self).__name__.lower()
        self.func = func
        self.__dict__.update(kwargs)
        # Cache the compiled functions, so they don't have to be recompiled on every call
        self._jit_scalar = self.callable(self.nans, self.reverse, scalar=True)
        self._jit_non_scalar = self.callable(self.nans, self.reverse, scalar=False)

    def __call__(self, group_idx, a, size=None, fill_value=0, order='C',
                 dtype=None, axis=None, ddof=0, x=None):
        iv = input_validation(group_idx, a, size=size, order=order,
                              axis=axis, check_bounds=False, func=self.func)
        group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

        # TODO: The typecheck should be done by the class itself, not by check_dtype
        dtype = check_dtype(dtype, self.func, a, len(group_idx))
        check_fill_value(fill_value, dtype, func=self.func)
        input_dtype = type(a) if np.isscalar(a) else a.dtype
        ret, counter, mean, outer = self._initialize(
            flat_size, fill_value, dtype, input_dtype, group_idx.size
        )
        group_idx = np.ascontiguousarray(group_idx)

        if not np.isscalar(a):
            a = np.ascontiguousarray(a)
            jitfunc = self._jit_non_scalar
        else:
            jitfunc = self._jit_scalar
        jitfunc(group_idx, a, ret, counter, mean, outer, fill_value, ddof, x)
        self._finalize(ret, counter, fill_value)

        if self.outer:
            ret = outer

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            if unravel_shape is not None:
                # argreductions only
                ret = np.unravel_index(ret, unravel_shape)[axis]
            ret = ret.reshape(size, order=order)
        return ret

    @classmethod
    def _initialize(cls, flat_size, fill_value, dtype, input_dtype, input_size):
        if cls.forced_fill_value is None:
            ret = np.full(flat_size, fill_value, dtype=dtype)
        else:
            ret = np.full(flat_size, cls.forced_fill_value, dtype=dtype)

        counter = mean = outer = None
        if cls.counter_fill_value is not None:
            counter = np.full_like(ret, cls.counter_fill_value, dtype=cls.counter_dtype)
        if cls.mean_fill_value is not None:
            dtype = cls.mean_dtype if cls.mean_dtype else input_dtype
            mean = np.full_like(ret, cls.mean_fill_value, dtype=dtype)
        if cls.outer:
            outer = np.full(input_size, fill_value, dtype=dtype)

        return ret, counter, mean, outer

    @classmethod
    def _finalize(cls, ret, counter, fill_value):
        if cls.forced_fill_value is not None and fill_value != cls.forced_fill_value:
            if cls.counter_dtype == bool:
                ret[counter] = fill_value
            else:
                ret[~counter.astype(bool)] = fill_value

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        """ Compile a jitted function doing the hard part of the job """
        _valgetter = cls._valgetter_scalar if scalar else cls._valgetter
        valgetter = nb.njit(_valgetter)
        outersetter = nb.njit(cls._outersetter)

        _cls_inner = nb.njit(cls._inner)
        if nans:
            def _inner(ri, val, ret, counter, mean):
                if not np.isnan(val):
                    _cls_inner(ri, val, ret, counter, mean)
            inner = nb.njit(_inner)
        else:
            inner = _cls_inner

        def _loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # fill_value and ddof need to be present for being exchangeable with loop_2pass
            size = len(ret)
            rng = range(len(group_idx) - 1, -1, -1) if reverse else range(len(group_idx))
            for i in rng:
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                val = valgetter(a, i)
                inner(ri, val, ret, counter, mean)
                outersetter(outer, i, ret[ri])
        return nb.njit(_loop, nogil=True)

    @staticmethod
    def _valgetter(a, i):
        return a[i]

    @staticmethod
    def _valgetter_scalar(a, i):
        return a

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        raise NotImplementedError("subclasses need to overwrite _inner")

    @staticmethod
    def _outersetter(outer, i, val):
        pass


class AggregateTrapz(AggregateOp):
    """Base class for the trapezoidal rule."""
    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        """ Compile a jitted function doing the hard part of the job """
        _valgetter = cls._valgetter_scalar if scalar else cls._valgetter
        valgetter = nb.njit(_valgetter)
        outersetter = nb.njit(cls._outersetter)

        _cls_inner = nb.njit(cls._inner)
        if nans:
            def _inner(ri, val, ret, counter, mean):
                if not np.isnan(val):
                    _cls_inner(ri, val, ret, counter, mean)
            inner = nb.njit(_inner)
        else:
            inner = _cls_inner

        def _loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof, x):
            # fill_value and ddof need to be present for being exchangeable with loop_2pass
            size = len(ret)
            rng = range(len(group_idx) - 2, -1, -1) if reverse else range(len(group_idx) - 1)
            for i in rng:
                ri = group_idx[i]
                # If both this index and next index are in same group:
                if group_idx[i] == group_idx[i+1]:
                    if ri < 0:
                        raise ValueError("negative indices not supported")
                    if ri >= size:
                        raise ValueError("one or more indices in group_idx are too large")
                    avg_y = (valgetter(a, i) + valgetter(a, i + 1)) / 2
                    if x is not None:
                        dx = valgetter(x, i + 1) - valgetter(x, i)
                    else:
                        dx = 1
                    val = avg_y * dx
                    inner(ri, val, ret, counter, mean, ddof)
                    outersetter(outer, i, ret[ri])
        return nb.njit(_loop, nogil=True)


class Trapz(AggregateTrapz):
    mean_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, ddof):
        ret[ri] += val


def get_funcs():
    funcs = dict()
    for op in (Sum, Prod, Len, All, Any, Last, First, AllNan, AnyNan, Min, Max,
               ArgMin, ArgMax, Mean, Std, Var,
               CumSum, CumProd, CumMax, CumMin, Trapz):
        funcname = op.__name__.lower()
        funcs[funcname] = op(funcname)
        if funcname not in funcs_no_separate_nan:
            funcname = 'nan' + funcname
            funcs[funcname] = op(funcname, nans=True)
    return funcs


class AggregateGeneric(AggregateOp):
    """Base class for jitting arbitrary functions."""
    counter_fill_value = None


def aggregate(group_idx, a, func='trapz', size=None, fill_value=0, order='C',
              dtype=None, axis=None, cache=None, x=None):
    import numpy_groupies
    if 'trapz' not in numpy_groupies.utils.funcs_common:
        numpy_groupies.utils.funcs_common.append("trapz")
    numpy_groupies.utils_numpy._alias_numpy.update({np.trapz: 'trapz'})
    _impl_dict = get_funcs()
    aliasing = get_aliasing(numpy_groupies.utils_numpy._alias_numpy)

    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        if cache in (None, False):
            aggregate_op = AggregateGeneric(func)
        else:
            if cache is True:
                cache = _default_cache
            aggregate_op = cache.setdefault(func, AggregateGeneric(func))
        return aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, x)
    else:
        func = _impl_dict[func]
        return func(group_idx, a, size, fill_value, order, dtype, axis, x=x)


def _binned_agg(
    array: np.ndarray,
    indices: np.ndarray,
    num_bins: int,
    *,
    func,
    fill_value,
    dtype,
) -> np.ndarray:
    """NumPy helper function for aggregating over bins, from @shoyer,
    see https://nbviewer.org/gist/shoyer/6d6c82bbf383fb717cc8631869678737"""
    mask = np.logical_not(np.isnan(indices))
    int_indices = indices[mask].astype(int)
    result = aggregate(
        int_indices, array[..., mask],
        func=func,
        size=num_bins,
        fill_value=fill_value,
        dtype=dtype,
        axis=-1,
    )
    return result


def groupby_bins_agg(
    array: xarray.DataArray,
    group: xarray.DataArray,
    bins,
    func='trapz',
    fill_value=0,
    dtype=None,
    **cut_kwargs,
) -> xarray.DataArray:
    """Faster equivalent of Xarray's groupby_bins(...).sum().
    Courtesy of @shoyer, https://nbviewer.org/gist/shoyer/6d6c82bbf383fb717cc8631869678737
    """
    # TODO: implement this upstream in xarray:
    # https://github.com/pydata/xarray/issues/4473
    binned = pd.cut(np.ravel(group), bins, **cut_kwargs)
    new_dim_name = group.name + "_bins"
    indices = group.copy(data=binned.codes.reshape(group.shape))
    result = xarray.apply_ufunc(
        _binned_agg, array, indices,
        input_core_dims=[indices.dims, indices.dims],
        output_core_dims=[[new_dim_name]],
        output_dtypes=[array.dtype],
        dask_gufunc_kwargs=dict(
            output_sizes={new_dim_name: binned.categories.size},
            # allow_rechunk=True
        ),
        kwargs={
            'num_bins': binned.categories.size,
            'func': func,
            'fill_value': fill_value,
            'dtype': dtype,
        },
        dask='parallelized',
    )
    result.coords[new_dim_name] = [
        0.5 * (b.left + b.right) for b in binned.categories
    ]
    return result.rename({new_dim_name: 'wavelength'})
