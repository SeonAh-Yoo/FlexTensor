import time
import argparse
import timeit
import torch
from flextensor.configs.conv2d_config import *
torch.backends.cudnn.enabled = False


shape_dict = {
    "yolo": yolo_shapes,
    "google": google_shapes,
    "squeeze": squeeze_shapes,
    "res": res_shapes,
    "vgg-16": vgg_16_shapes,
    "vgg-19": vgg_19_shapes
}


def pytorch_cpu(batch_size, height, width, channel, kernel_size, output_channel, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
    run_time = timeit.timeit(setup= 'import torch\n'
                                    'conv = torch.nn.functional.conv_transpose2d\n'
                                    'A = torch.rand([' + str(batch_size) + ', ' + str(channel) + ', ' + str(height) + ', ' + str(width) + '], dtype=torch.float32)\n'
                                    'W = torch.rand([' + str(channel) + ', ' + str(output_channel//groups) + ', ' + str(kernel_size) + ', ' + str(kernel_size) + '], dtype=torch.float32)\n'
                                    'conv(A, W, stride=' + str(stride) + ', padding=' + str(padding) + ', dilation=' + str(dilation) + ', groups=' + str(groups) + ')\n',
                               stmt='ans = conv(A, W, stride=' + str(stride) + ', padding=' + str(padding) + ', dilation=' + str(dilation) + ', groups=' + str(groups) + ')',
                               number=number)
    return run_time / number * 1e3


def pytorch_cuda(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
    A = torch.rand([N, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    W = torch.rand([C, K//groups, kernel_size, kernel_size], dtype=torch.float32).cuda("cuda:" + str(dev))

    # warm-up
    torch.nn.functional.conv_transpose2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    torch.cuda.synchronize()
    sum_time = 0.0
    for i in range(number):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ans = torch.nn.functional.conv_transpose2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        sum_time += start.elapsed_time(end)
    return sum_time / number


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shapes", help="Use which shapes [yolo, google, res, squeeze, vgg-16, vgg-19]", type=str, default="yolo")
    parser.add_argument("-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape", type=int, default=-1)
    parser.add_argument("-n", "--number", help="number test run", type=int, default=10)
    parser.add_argument("--target", help="target device type", type=str, default="llvm")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--type", help="type of baseline", type=str, default="pytorch")

    args = parser.parse_args()
    shapes = shape_dict[args.shapes]
    if args.to < 0:
        end = len(shapes)
    else:
        end = args.to
    shapes = shapes[args.from_:end]
    if args.type == "pytorch":
        if args.target == "cuda":
            baseline = pytorch_cuda
        elif args.target == "llvm":
            baseline = pytorch_cpu
        else:
            raise RuntimeError("Only support target 'llvm' and 'cuda', but got %s"%args.target)
    else:
        raise RuntimeError("Only implement pytorch baseline now, no '%s' baseline"%args.type)
    
    print("%s baselines for %s convolution 2d for target %s (%d):" % (args.type, args.shapes, args.target, args.device))
    for i, shape in enumerate(shapes):
        count = i + args.from_ 
        print("layer", count)
        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
        rout_channel = in_channel
        rin_channel = out_channel
        rheight = (height + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
        rwidth = (width + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
        cost = baseline(batch, rheight, rwidth, rin_channel, k_h, rout_channel, stride=stride, padding=padding, dilation=dilation, groups=groups, number=args.number, dev=args.device)
        print("Use %f(ms)" % cost)
    print("Done!")
