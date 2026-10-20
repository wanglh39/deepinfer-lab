// miniflow._C —— 算子层 C++ 后端（对应 DeepInfer 的 FlashMLA/DeepGEMM 算子库）
//
// DeepInfer 的关键认知：推理框架的性能上限由算子决定，引擎层的调度与
// 并行都是为了让算子「吃饱」。FlashMLA（MLA 注意力核）、DeepGEMM
// （矩阵乘）、DeepEP（MoE 通信）每一块都是为特定负载手写的核。
//
// 本文件是教学版，用三个最小例子复刻「算子层为什么值得单独一讲」：
//
//   1. matmul_naive vs matmul —— 同样的 O(M*K*N)，ijk 内积视角与
//      ikj 外积视角差一个量级：循环序决定 b 的访问是否按行连续
//      （缓存局部性）。DeepGEMM 的分块/向量化是同一思想的工程极致；
//
//   2. fused_silu_swiglu —— 三个算子（silu、mul、内存往返）融合成
//      一个核。生产框架的融合发生在编译期（TileLang 生成的核），
//      收益是省掉中间张量的显存带宽 —— decode 阶段访存受限，
//      省带宽就是省时间；
//
//   3. softmax/rms_norm —— 数值稳定的标准实现，对应 FlashMLA 里
//      的在线 softmax（分块重整化，避免物化整个注意力矩阵）。
//
// Python 侧由 miniflow/tensor.py 派发：探测到本模块则 BACKEND="cpp"，
// 否则回落 NumPy —— 对应生产框架「多算子后端按硬件自动选择」。

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using f32 = float;

using cf32 = py::array_t<f32, py::array::c_style | py::array::forcecast>;

// ----------------------------------------------------------------------
// 矩阵乘：一切的开端（DeepGEMM 的领地）
// ----------------------------------------------------------------------
static py::array_t<f32> matmul_impl(const cf32 &a, const cf32 &b,
                                    bool naive) {
    py::buffer_info ai = a.request(), bi = b.request();
    if (ai.ndim != 2 || bi.ndim != 2)
        throw std::invalid_argument("matmul: 期望二维矩阵");
    const ssize_t M = ai.shape[0], K = ai.shape[1];
    const ssize_t K2 = bi.shape[0], N = bi.shape[1];
    if (K != K2)
        throw std::invalid_argument("matmul: 内维不匹配");

    const f32 *pa = static_cast<const f32 *>(ai.ptr);
    const f32 *pb = static_cast<const f32 *>(bi.ptr);
    auto out = py::array_t<f32>({M, N});
    py::buffer_info oi = out.request();
    f32 *po = static_cast<f32 *>(oi.ptr);
    std::fill(po, po + M * N, 0.0f);

    if (naive) {
        // ijk 内积视角：c[i][j] = <a 的第 i 行, b 的第 j 列>。
        // b 按列访问 => 每步都缓存 miss —— 教科书慢的根源。
        for (ssize_t i = 0; i < M; ++i)
            for (ssize_t j = 0; j < N; ++j) {
                f32 acc = 0.0f;
                for (ssize_t k = 0; k < K; ++k)
                    acc += pa[i * K + k] * pb[k * N + j];
                po[i * N + j] = acc;
            }
    } else {
        // ikj 外积视角：c[i] += a[i][k] * b[k]（整行广播累加）。
        // b 的第 k 行按内存序连续扫 —— 缓存友好的最小改造。
        for (ssize_t i = 0; i < M; ++i) {
            f32 *ci = po + i * N;
            const f32 *ai_row = pa + i * K;
            for (ssize_t k = 0; k < K; ++k) {
                const f32 aik = ai_row[k];
                if (aik == 0.0f) continue;
                const f32 *bk = pb + k * N;
                for (ssize_t j = 0; j < N; ++j)
                    ci[j] += aik * bk[j];
            }
        }
    }
    return out;
}

py::array_t<f32> matmul_naive(const cf32 &a, const cf32 &b) {
    return matmul_impl(a, b, true);
}

py::array_t<f32> matmul(const cf32 &a, const cf32 &b) {
    return matmul_impl(a, b, false);
}

// ----------------------------------------------------------------------
// softmax（最后一维）：注意力的心脏
// ----------------------------------------------------------------------
py::array_t<f32> softmax_lastdim(const cf32 &x) {
    py::buffer_info xi = x.request();
    if (xi.ndim < 1)
        throw std::invalid_argument("softmax: 至少一维");
    const ssize_t rows = xi.shape[0] == 0 ? 0 : xi.size / xi.shape[xi.ndim - 1];
    const ssize_t n = xi.shape[xi.ndim - 1];
    if (n == 0)
        throw std::invalid_argument("softmax: 空的归约维");

    auto out = py::array_t<f32>(xi.shape);
    const f32 *px = static_cast<const f32 *>(xi.ptr);
    f32 *po = static_cast<f32 *>(out.request().ptr);

    // 数值稳定两遍法：先 max 再 exp/sum。
    // FlashMLA 的在线 softmax 用分块重整化把两遍压成一趟 —— 同一数学。
    for (ssize_t r = 0; r < rows; ++r) {
        const f32 *xr = px + r * n;
        f32 *orow = po + r * n;
        f32 mx = *std::max_element(xr, xr + n);
        f32 sum = 0.0f;
        for (ssize_t j = 0; j < n; ++j) {
            f32 e = std::exp(xr[j] - mx);
            orow[j] = e;
            sum += e;
        }
        const f32 inv = 1.0f / sum;
        for (ssize_t j = 0; j < n; ++j) orow[j] *= inv;
    }
    return out;
}

// ----------------------------------------------------------------------
// RMSNorm（最后一维）：DeepSeek 系列全部用它
// ----------------------------------------------------------------------
py::array_t<f32> rms_norm_lastdim(const cf32 &x, const cf32 &weight,
                                  double eps) {
    py::buffer_info xi = x.request(), wi = weight.request();
    if (xi.ndim < 1 || wi.ndim != 1)
        throw std::invalid_argument("rms_norm: 形状不合法");
    const ssize_t n = xi.shape[xi.ndim - 1];
    if (wi.shape[0] != n)
        throw std::invalid_argument("rms_norm: weight 长度不匹配");
    const ssize_t rows = xi.size / n;

    auto out = py::array_t<f32>(xi.shape);
    const f32 *px = static_cast<const f32 *>(xi.ptr);
    const f32 *pw = static_cast<const f32 *>(wi.ptr);
    f32 *po = static_cast<f32 *>(out.request().ptr);

    for (ssize_t r = 0; r < rows; ++r) {
        const f32 *xr = px + r * n;
        f32 *orow = po + r * n;
        f32 ms = 0.0f;
        for (ssize_t j = 0; j < n; ++j) ms += xr[j] * xr[j];
        const f32 inv = 1.0f / std::sqrt(ms / static_cast<f32>(n)
                                         + static_cast<f32>(eps));
        for (ssize_t j = 0; j < n; ++j) orow[j] = xr[j] * inv * pw[j];
    }
    return out;
}

// ----------------------------------------------------------------------
// SwiGLU 融合核：三个算子一个核（TileLang 融合的最小教学版）
// ----------------------------------------------------------------------
py::array_t<f32> fused_silu_swiglu(const cf32 &gate, const cf32 &up) {
    py::buffer_info gi = gate.request(), ui = up.request();
    if (gi.ndim != 2 || ui.ndim != 2 || gi.shape[0] != ui.shape[0]
        || gi.shape[1] != ui.shape[1])
        throw std::invalid_argument("fused_silu_swiglu: gate/up 形状不匹配");
    const ssize_t total = gi.size;

    auto out = py::array_t<f32>(gi.shape);
    const f32 *pg = static_cast<const f32 *>(gi.ptr);
    const f32 *pu = static_cast<const f32 *>(ui.ptr);
    f32 *po = static_cast<f32 *>(out.request().ptr);

    // act = silu(gate) * up，单趟完成：
    // 朴素实现是「silu(gate) 临时张量 -> mul -> 再临时」两度访存；
    // 融合核只有一次读入一次写出 —— decode 访存受限，省带宽即省时间。
    for (ssize_t i = 0; i < total; ++i) {
        const f32 g = pg[i];
        const f32 sig = 1.0f / (1.0f + std::exp(-g));
        po[i] = g * sig * pu[i];
    }
    return out;
}

// ----------------------------------------------------------------------
PYBIND11_MODULE(_C, m) {
    m.doc() = "miniflow 算子层 C++ 后端（对应 DeepInfer 的手写算子库）";
    m.def("matmul", &matmul, py::arg("a"), py::arg("b"),
          "矩阵乘（ikj 缓存友好序）");
    m.def("matmul_naive", &matmul_naive, py::arg("a"), py::arg("b"),
          "矩阵乘（ijk 教科书序，教学基准）");
    m.def("softmax", &softmax_lastdim, py::arg("x"),
          "数值稳定 softmax（最后一维）");
    m.def("rms_norm", &rms_norm_lastdim, py::arg("x"), py::arg("weight"),
          py::arg("eps") = 1e-5, "RMSNorm（最后一维）");
    m.def("fused_silu_swiglu", &fused_silu_swiglu, py::arg("gate"),
          py::arg("up"), "SwiGLU 融合核：silu(gate)*up 单趟完成");
}