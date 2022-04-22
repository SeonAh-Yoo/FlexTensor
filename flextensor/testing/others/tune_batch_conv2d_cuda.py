"""
Tune batched convolution with NCHW layout
Target x86 CPU
The contents refer to TVM tutorials

====================================
**Author**: `Size Zheng`
"""

import logging
import sys
import numpy as np

import tvm
from flextensor.nn import conv2d_nchw

from tvm import autotvm

# to run this, first:
# export PATH=/usr/local/cuda-10.1/nvvm/libdevice:$PATH

@autotvm.template
def conv2d_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    data = tvm.te.placeholder((N, CI, H, W), name='data', dtype="float32")
    kernel = tvm.te.placeholder((CO, CI, KH, KW), name='kernel', dtype="float32")
    conv = conv2d_nchw(data, kernel, stride=stride, padding=padding)
    s = tvm.te.create_schedule([conv.op])

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    fused = s[conv].fuse(y, x)
    cfg = autotvm.get_config()
    cfg.define_split("tile_n", n, num_outputs=4)
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, 'local')

    # create cache stage
    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])
    AL = s.cache_read(AA, 'local', [OL])
    WL = s.cache_read(WW, 'local', [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    yx = s[output].fuse(y, x)
    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    kernel_scope = yx

    s[output].bind(yx, tvm.te.thread_axis("blockIdx.z"))
    s[output].bind(bn, tvm.te.thread_axis("blockIdx.y"))
    s[output].bind(bf, tvm.te.thread_axis("blockIdx.x"))
    s[output].bind(vn, tvm.te.thread_axis("vthread"))
    s[output].bind(vf, tvm.te.thread_axis("vthread"))
    s[output].bind(tn, tvm.te.thread_axis("threadIdx.y"))
    s[output].bind(tf, tvm.te.thread_axis("threadIdx.x"))
    s[output].reorder(yx, bn, bf, vn, vf, tn, tf, ni, fi)
    s[OL].compute_at(s[output], tf)

    # tile reduction axes
    n, f, yx = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, rym, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxm, rxi = cfg['tile_ry'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, yx)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        ty, fused = s[load].split(fused, nparts=cfg["tile_n"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        s[load].bind(ty, tvm.te.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.te.thread_axis("threadIdx.x"))

    # tune unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]

######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# logging config (for printing tuning log to screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


# the last layer in yolo
def run(name, N, H, W, CO, CI, KH, KW, stride, pad):
    N, H, W, CO, CI, KH, KW, strides, padding = N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad)
    task = autotvm.task.create(conv2d_batching,
                               args=(N, H, W, CO, CI, KH, KW, strides, padding),
                               target='cuda')
    print(task.config_space)
    logfile = "conv2d_" + name + ".log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=200,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(logfile)])

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(logfile)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(logfile):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_batching(N, H, W, CO, CI, KH, KW, strides, padding)
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty((N, CO, (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1), ctx=ctx)
    # func(a_tvm, w_tvm, c_tvm)

    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    cost = evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3
    print('Time cost of this operator: %f' % cost)
    with open("autotvm_conv_nchw.txt", "a") as f:
        f.write("name, {}\n".format(cost))


if __name__ == "__main__":
    arg_lst = [
        (1, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (8, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (64, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (256, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (1, 14, 14, 1024, 1, 1, 512, 1, 0),
        # (1, 28, 28, 256, 3, 3, 512, 1, 1),
        # (1, 28, 28, 512, 1, 1, 256, 1, 0),
        # (1, 56, 56, 128, 3, 3, 256, 1, 1),
        # (1, 56, 56, 192, 1, 1, 128, 1, 0),
        # (1, 112, 112, 64, 3, 3, 192, 1, 1),
        # (1, 448, 448, 3, 7, 7, 64, 2, 3)
    ]
    names = [
        "yolo24_b1",
        # "yolo24_b8",
        # "yolo24_b64",
        # "yolo24_b256",
        # "yolo19_b1",
        # "yolo10_b1",
        # "yolo7_b1",
        # "yolo4_b1",
        # "yolo3_b1",
        # "yolo2_b1",
        # "yolo1_b1"
    ]
    for i in range(len(arg_lst)):
        name = names[i]
        N, H, W, CI, KW, KH, CO, stride, pad = arg_lst[i]
        run(name, N, H, W, CO, CI, KH, KW, stride, pad)

