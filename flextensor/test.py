import tvm
from flextensor.task import register_task, Task
from flextensor.scheduler import schedule

def gemm(A, B):
    """Matrix multiplies matrix 

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [height, width]
    B: tvm.tensor.Tensor
        shape [width, length]
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [height, length]
    -----------------------------
    """
    k = tvm.te.reduce_axis((0, B.shape[0]))
    return tvm.te.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k))

def wrap_gemm(N, K, M):
    A = tvm.te.placeholder((N, K))
    B = tvm.te.placeholder((K, M))
    Output = gemm(A, B)
    return [Output.op], [A, B, Output]

'''
To create a task, the parameters are:
1. type of operator: str
2. name of this operator: str
3. the wrapper for tensor computation
4. arguments to the wrapper, i.e. input shapes
5. target device: str ("llvm" or "cuda" currently)
6. device number: int
'''
task = Task(
    "gemm", 
    "gemm", 
    wrap_gemm, 
    (1024, 1024, 1024), 
    # "llvm",
    "cuda", 
    0)
# register the task
register_task(task)

s, bufs, configs = schedule(
            task.key, # give the key of target task
            slevel=4,
            rlevel=3,
            # op_trial=100,
            op_trial=3, 
            timeout=10, 
            op_stop=30, 
            method="nns", 
            parallel=1,
            use_model=True,
            )

# directly use the results
func = tvm.build(s, bufs, task.target)
# use the configs
# from flextensor.scheduler import schedule_with_config

# s, bufs = schedule_with_config(task_key, configs)
# func = tvm.build(s, bufs, task.target)





# import tvm
# import numpy as np
# from tvm import te
# from tempfile import mkstemp, mkdtemp
# import os

# # 1. Define GEMM computation
# M, K, N = 1024, 1024, 1024
# A = te.placeholder((M, K), name="A")
# B = te.placeholder((K, N), name="B")
# k = te.reduce_axis((0, K), name="k")
# C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# # 2. Schedule and build
# s = te.create_schedule(C.op)
# target = tvm.target.Target("llvm")
# func = tvm.build(s, [A, B, C], target=target, name="gemm")
# lib_dir = mkdtemp(prefix="bb_lib_")
# fd, lib = mkstemp(prefix="bbb_builtin_function_", suffix=".so",
#                                       dir=lib_dir)
# os.close(fd)
# print("aaa")
# func.export_library(lib, fcompile=None)
# print("bbb")

# # 3. Prepare input data
# ctx = tvm.cpu(0)
# a_np = np.random.uniform(size=(M, K)).astype("float32")
# b_np = np.random.uniform(size=(K, N)).astype("float32")
# c_np = np.zeros((M, N), dtype="float32")
# a_tvm = tvm.nd.array(a_np, ctx)
# b_tvm = tvm.nd.array(b_np, ctx)
# c_tvm = tvm.nd.array(c_np, ctx)

# # 4. Time evaluator
# evaluator = func.time_evaluator(func.entry_name, ctx, number=10, repeat=5)
# time_results = evaluator(a_tvm, b_tvm, c_tvm)

# # 5. Print performance results
# print(f"Mean execution time: {time_results.mean * 1e3:.2f} ms")
# print(f"Median execution time: {time_results.median * 1e3:.2f} ms")
# print(f"Standard deviation: {time_results.std * 1e3:.2f} ms")