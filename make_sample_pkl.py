#!/usr/bin/env python3
"""
构造一个示例 CompileResult pkl 文件，用于测试 cmp.py。
参考 triton_npu_trace_readable.txt 的格式构建。
"""

import pickle
import sys
from pathlib import Path

# 让 src 目录可导入
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ttadapter_diff.schema import CompileResult, TritonKernelMetadata

# ---- 构造一段模拟的 ttadapter MLIR 文本（含 func.func 和 loc 引用）----
SAMPLE_TTADAPTER_STR = r"""#loc = loc("/data/c00961524/flash_attention_npu_v8.py":1108:0)
#loc1 = loc("/data/c00961524/flash_attention_npu_v8.py":1110:0)
module {
  func.func @bwd_preprocess_ifmn(%arg0: !tt.ptr<bf16, 1> {tt.divisibility = 16 : i32} loc(#loc), %arg1: !tt.ptr<bf16, 1> {tt.divisibility = 16 : i32} loc(#loc), %arg2: !tt.ptr<fp32, 1> {tt.divisibility = 16 : i32} loc(#loc)) -> i32 attributes {tt.func_arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}], tt.maxntid = 32 : i32} {
    %0 = tt.make_tensor_descriptor %arg0 shape = (32 : i32) strides = (1 : i32) element_type = bf16 loc(#loc1)
    %1 = tt.make_tensor_descriptor %arg1 shape = (32 : i32) strides = (1 : i32) element_type = bf16 loc(#loc1)
    %2 = tt.load %0 : !tt.descr<bf32, 1> loc(#loc1)
    %3 = tt.load %1 : !tt.descr<bf16, 1> loc(#loc1)
    %4 = arith.mulf %2, %3 : f32 loc(#loc1)
    tt.store %arg2, %4 : f32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    tt.return %c0_i32 : i32 loc(#loc)
  } loc(#loc)
} loc(#loc)
"""

# ---- 构造元数据 ----
meta = TritonKernelMetadata(
    kernel_name="bwd_preprocess_ifmn",
    module_name="flash_attention_npu_v8",
    kernel_path=Path("/data/c00961524/flash_attention_npu_v8.py"),
    rel_kernel_path=Path("flash_attention_npu_v8.py"),
    signature={
        "o_ptr": "*bf16",
        "do_ptr": "*bf16",
        "d_ptr": "*fp32",
    },
    constants={
        "Q_HEAD_NUM": 8,
        "V_DIM": 64,
        "DTYPE": 14,
        "TASK_SIZE": 46,
        "NUM_BLOCKS": 2,
        "BLOCK_SIZE": 32,
    },
    # target 用 dict 代替 triton.GPUTarget（schema 中类型为 Any）
    target={"backend": "npu", "arch": "Ascend950PR_9579", "warp_size": 0},
    options={
        "debug": False,
        "num_warps": 32,
        "num_ctas": 1,
        "num_stages": 2,
        "warp_size": 32,
    },
    call_stack=[
        'File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/autograd/function.py", line 307, in apply\n    return user_fn(self, *args)',
        'File "/data/c00961524/flash_attention_npu_v8.py", line 1108, in backward\n    bwd_preprocess_ifmn[(NUM_CORES,)](',
    ],
    sys_argv=["/usr/local/python3.11.13/bin/pytest", "-sv", "/data/c00961524/flash_attention_npu_v8.py"],
)

result = CompileResult(ttadapter_str=SAMPLE_TTADAPTER_STR, meta=meta)

out_path = Path(__file__).parent / "sample_trace.pkl"
with out_path.open("wb") as f:
    pickle.dump(result, f)

print(f"已生成: {out_path}")
print(f"  ttadapter_str 长度: {len(SAMPLE_TTADAPTER_STR)}")
print(f"  kernel_name: {meta.kernel_name}")
