"""Integer dot products of 8- and 4-bit codes with int8 query weights.

The quantized search can score candidates as ``sum_j w_j * c_j`` with the query
folded into int8 weights ``w`` and ``c`` the unsigned codes. These intrinsics
compute that sum 64 bytes at a time: with AVX-512 VNNI as ``vpdpbusd`` (64
u8 x s8 products per instruction); elsewhere as portable vector IR that LLVM
lowers to the best the target offers. Code rows and query rows must be padded
(with zeros) to a multiple of ``CODE_ALIGNMENT`` bytes.

For 4-bit codes byte ``t`` packs dimension ``2t`` in its low nibble and
``2t + 1`` in its high nibble; the query is then a (2, n_bytes) array of the
weights for the low and the high nibbles.
"""
import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils, config
from numba.extending import intrinsic

CODE_ALIGNMENT = 64

_I8 = ir.IntType(8)
_I32 = ir.IntType(32)
_V64I8 = ir.VectorType(_I8, 64)
_V32I16 = ir.VectorType(ir.IntType(16), 32)
_V16I32 = ir.VectorType(_I32, 16)
_V64I32 = ir.VectorType(_I32, 64)


def _target_has_vnni():
    """Whether numba compiles for a CPU with AVX-512 VNNI (and BW)."""
    cpu_name = config.CPU_NAME
    if cpu_name and cpu_name != "host":
        features = (config.CPU_FEATURES or "").split(",")
        return "+avx512vnni" in features and "+avx512bw" in features
    try:
        import llvmlite.binding as llvm_binding

        host = llvm_binding.get_host_cpu_features()
        return bool(host.get("avx512vnni", False)) and bool(host.get("avx512bw", False))
    except Exception:
        return False


TARGET_HAS_VNNI = _target_has_vnni()


def aligned_padded_codes(codes):
    """``codes`` with rows zero-padded to a multiple of ``CODE_ALIGNMENT`` bytes,
    starting on a ``CODE_ALIGNMENT`` boundary (``codes`` itself if it already is)."""
    n, width = codes.shape
    padded_width = -(-width // CODE_ALIGNMENT) * CODE_ALIGNMENT
    if (
        padded_width == width
        and codes.flags.c_contiguous
        and codes.dtype == np.uint8
        and codes.ctypes.data % CODE_ALIGNMENT == 0
    ):
        return codes
    buffer = np.empty(n * padded_width + CODE_ALIGNMENT, dtype=np.uint8)
    offset = -buffer.ctypes.data % CODE_ALIGNMENT
    out = buffer[offset : offset + n * padded_width].reshape(n, padded_width)
    block = 65536
    for start in range(0, n, block):
        out[start : start + block, :width] = codes[start : start + block]
    out[:, width:] = 0
    return out


def _is_codes(codes):
    return isinstance(codes, types.Array) and codes.ndim == 2 and codes.dtype == types.uint8


def _is_query(query):
    return isinstance(query, types.Array) and query.ndim == 2 and query.dtype == types.int8


def _row_pointer(context, builder, array_type, array_value, row_value, row_type):
    array = context.make_array(array_type)(context, builder, array_value)
    row = context.cast(builder, row_value, row_type, types.intp)
    zero = context.get_constant(types.intp, 0)
    pointer = cgutils.get_item_pointer(
        context, builder, array_type, array, [row, zero], wraparound=False
    )
    return builder.bitcast(pointer, _I8.as_pointer())


def _load_chunk(builder, base, offset):
    pointer = builder.bitcast(builder.gep(base, [offset]), _V64I8.as_pointer())
    return builder.load(pointer, align=1)


class _Accumulator:
    """A running ``sum_j c_j * w_j`` over 64-byte chunks (c unsigned, w signed)."""

    def __init__(self, builder):
        self.builder = builder
        if TARGET_HAS_VNNI:
            fnty = ir.FunctionType(_V16I32, [_V16I32, _V16I32, _V16I32])
            self.dpbusd = cgutils.get_or_insert_function(
                builder.module, fnty, "llvm.x86.avx512.vpdpbusd.512"
            )
            self.slot = cgutils.alloca_once_value(builder, ir.Constant(_V16I32, None))
        else:
            fnty = ir.FunctionType(_I32, [_V64I32])
            self.reduce = cgutils.get_or_insert_function(
                builder.module, fnty, "llvm.vector.reduce.add.v64i32"
            )
            self.slot = cgutils.alloca_once_value(builder, ir.Constant(_I32, 0))

    def add(self, codes, weights):
        builder = self.builder
        if TARGET_HAS_VNNI:
            value = builder.call(
                self.dpbusd,
                [
                    builder.load(self.slot),
                    builder.bitcast(codes, _V16I32),
                    builder.bitcast(weights, _V16I32),
                ],
            )
        else:
            products = builder.mul(builder.zext(codes, _V64I32), builder.sext(weights, _V64I32))
            value = builder.add(builder.load(self.slot), builder.call(self.reduce, [products]))
        builder.store(value, self.slot)

    def total(self):
        value = self.builder.load(self.slot)
        if TARGET_HAS_VNNI:
            fnty = ir.FunctionType(_I32, [_V16I32])
            reduce = cgutils.get_or_insert_function(
                self.builder.module, fnty, "llvm.vector.reduce.add.v16i32"
            )
            return self.builder.call(reduce, [value])
        return value


def _emit_dots(context, builder, sig, args, n_rows, nibbles):
    codes_type, query_type = sig.args[0], sig.args[-1]
    row_pointers = [
        _row_pointer(context, builder, codes_type, args[0], args[1 + r], sig.args[1 + r])
        for r in range(n_rows)
    ]
    query = context.make_array(query_type)(context, builder, args[-1])
    width = builder.extract_value(query.shape, 1)
    query_rows = [
        _row_pointer(
            context, builder, query_type, args[-1], context.get_constant(types.intp, h),
            types.intp,
        )
        for h in range(2 if nibbles else 1)
    ]
    accumulators = [
        [_Accumulator(builder) for _ in range(2 if nibbles else 1)] for _ in range(n_rows)
    ]
    mask = ir.Constant(_V64I8, [15] * 64)
    shift = ir.Constant(_V32I16, [4] * 32)
    n_chunks = builder.lshr(width, ir.Constant(width.type, 6))
    with cgutils.for_range(builder, n_chunks) as loop:
        offset = builder.shl(loop.index, ir.Constant(width.type, 6))
        weights = [_load_chunk(builder, q, offset) for q in query_rows]
        for r in range(n_rows):
            chunk = _load_chunk(builder, row_pointers[r], offset)
            if nibbles:
                low = builder.and_(chunk, mask)
                high = builder.and_(
                    builder.bitcast(builder.lshr(builder.bitcast(chunk, _V32I16), shift), _V64I8),
                    mask,
                )
                accumulators[r][0].add(low, weights[0])
                accumulators[r][1].add(high, weights[1])
            else:
                accumulators[r][0].add(chunk, weights[0])
    totals = []
    for r in range(n_rows):
        total = accumulators[r][0].total()
        if nibbles:
            total = builder.add(total, accumulators[r][1].total())
        totals.append(total)
    if n_rows == 1:
        return totals[0]
    return context.make_tuple(builder, sig.return_type, totals)


def _typer(n_rows, nibbles):
    def typer(typingctx, codes, *rest):
        rows, query = rest[:-1], rest[-1]
        if len(rows) != n_rows or not _is_codes(codes) or not _is_query(query):
            return None
        if not all(isinstance(r, types.Integer) for r in rows):
            return None
        return_type = types.int32 if n_rows == 1 else types.UniTuple(types.int32, n_rows)
        sig = return_type(codes, *rows, query)

        def codegen(context, builder, sig, args):
            return _emit_dots(context, builder, sig, args, n_rows, nibbles)

        return sig, codegen

    return typer


@intrinsic
def dot_u8(typingctx, codes, row, query):
    """``sum_j codes[row, j] * query[0, j]`` as int32."""
    return _typer(1, False)(typingctx, codes, row, query)


@intrinsic
def dot4_u8(typingctx, codes, row0, row1, row2, row3, query):
    """``dot_u8`` for four rows at once (sharing the query loads)."""
    return _typer(4, False)(typingctx, codes, row0, row1, row2, row3, query)


@intrinsic
def dot_u4(typingctx, codes, row, query):
    """``sum_t low(codes[row, t]) * query[0, t] + high(codes[row, t]) * query[1, t]``."""
    return _typer(1, True)(typingctx, codes, row, query)


@intrinsic
def dot4_u4(typingctx, codes, row0, row1, row2, row3, query):
    """``dot_u4`` for four rows at once (sharing the query loads)."""
    return _typer(4, True)(typingctx, codes, row0, row1, row2, row3, query)
