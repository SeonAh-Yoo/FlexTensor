import os
# import graph as _graph
import torch
import torch.multiprocessing as _multi
multi = _multi.get_context("spawn")
import tvm
import copy
import math
import json
import flextensor.utils as utils
import flextensor.model as model
import flextensor.space as space
import flextensor.task as task
import numpy as np
import logging
import argparse
import signal
import psutil


from collections import namedtuple, deque
try:
    from flextensor.scheduler import schedule_with_config_ops
    _has_flextensor = True
except ImportError:
    print("Can't import flextensor, please install it from https://github.com/KnowingNothing/FlexTensor.git")
    print("No performance test")
    _has_flextensor = False
    

TargetItem = namedtuple("TargetItem", "target dev_id")
DataItem = namedtuple("DataItem", "op shape target config cost")
ProcessedItem = namedtuple("ProcessedItem", "superschedule cost")


def load_data(train_file, test_file, eval_dev=-1):
    for filename in [train_file, test_file]:
        if filename is None:
            continue
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            raise RuntimeError("File not found when loading data from %s" % filename)

    def _load(filename):
        if filename is None:
            return []
        ret = []
        with open(filename, "r") as fin:
            for line in fin:
                if line:
                    key, string = line.split(":", 1)
                    string, cost = string.rsplit(":", 1)
                    op, _, shape_str, target_str = key.split("_")
                    shape = [int(x) for x in shape_str[1:-1].split(", ")]
                    shape = tuple(shape)
                    target, dev_id_str = target_str[:-1].split("(")
                    dev_id = int(dev_id_str) if eval_dev < 0 else eval_dev
                    config = json.loads(string)
                    ret.append(DataItem(
                        op=op, 
                        shape=shape, 
                        target=TargetItem(target=target, dev_id=dev_id), 
                        config=utils.Config(config[0], config[1]),
                        cost=float(cost))
                    )
        return ret
    
    return _load(train_file), _load(test_file)


# def to_tensor(data):
#     return torch.FloatTensor(data) / _graph.MAX_EXTENT


def preprocess(dataset, onehot=False):
    # ret = []
    ret = {}
    for data in dataset:
        # graph = get_graph(data.op, data.shape)
        tvm_op_lst, tvm_bufs = get_compute(data.op, data.shape)
        tvm_op = tvm_op_lst[-1]
        schedule_space = get_space(tvm_op)
        # schedule_choices = {}
        # label = {}
        superschedule = []
        for key, name_lst in schedule_space.types.items():
            # schedule_choices[key] = []
            # label[key] = []
            for i, name in enumerate(name_lst):
                # choices = schedule_space.subspaces[name].static_entities
                label_choice = data.config.op_config_lst[-1][key][i]
                if key == "fuse":
                    fuse_ = [1<<p for p in range(2)]
                    for l in label_choice:
                        superschedule.append(fuse_.index(int(l)))
                elif key == "spatial" or key == "reduce":
                    spa_red_ = [1<<p for p in range(11)]
                    for l in label_choice:
                        superschedule.append(spa_red_.index(int(l)))
                elif key == "reorder":
                    superschedule.extend(label_choice)
                elif key == "unroll":
                    unroll_ = [0, 1, 512, 1500]
                    for l in label_choice:
                        superschedule.append(unroll_.index(int(l)))
                # index = choices.index(label_choice)
                # if onehot:
                #     onehot_label = torch.zeros([len(choices)])
                #     onehot_label[index] = 1.0
                #     label[key].append(onehot_label)
                # else:
                #     label[key].append(torch.LongTensor([index]))
                # preprocessed = to_tensor(choices)
                # schedule_choices[key].append(preprocessed)
        # ret.append(ProcessedItem(raw=data, graph=graph, schedule_choices=schedule_choices, label=label))
        if not data.shape in ret.keys():
            ret[data.shape] = []
        ret[data.shape].append(ProcessedItem(superschedule=superschedule, cost=data.cost))
    
    return ret


def flatten_graph(ops):
    bfs_order = []
    down_graph = {}
    visited = set()
    q = deque()
    for op in ops:
        q.append(op)
        visited.add(op)
    while q:
        cur = q.popleft()
        if isinstance(cur, tvm.te.tensor.ComputeOp):
            bfs_order.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                visited.add(t.op)
                q.append(t.op)
            if t not in down_graph:
                down_graph[t] = []
            down_graph[t].append(cur)
    return list(reversed(bfs_order)), down_graph


def _evaluate(s, bufs, target, dev_id, number=1, q=None):
    ctx = tvm.device(target, dev_id)
    tvm_arys = []
    for arg in bufs:
        shape = utils.to_tuple(arg.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(arg.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    func, evaluator = None, None
    try:
        func = tvm.build(s, bufs, target)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean * 1e3
        if q:
            q.put(time_cost)
        return time_cost
    except Exception as e:
        for item in tvm_arys:
            del item
        if func is not None:
            del func
        if evaluator is not None:
            del evaluator
        raise e


def get_compute(op, shape):
    if op == "gemm":
        return task.gemm(*shape)
    elif op == "conv2d":
        return task.conv2d(*shape)
    else:
        raise RuntimeError("Not supported op compute type: %s" % op)


def evaluate(op_name, shape, config, target, dev_id, q):
    tvm_op_lst, tvm_bufs = get_compute(op_name, shape)
    s, bufs = schedule_with_config_ops(tvm_op_lst, tvm_bufs, config, target=target)
    try:
        dec_cost = _evaluate(s, bufs, target, dev_id, number=10)
    except Exception as e:
        dec_cost = str(e)
    q.put(dec_cost)


# def get_graph(op, shape):
#     if op == "gemm":
#         return _graph.graph_gemm(*shape)
#     elif op == "conv2d":
#         return _graph.conv2d_graph(*shape)
#     else:
#         raise RuntimeError("Not supported op graph type: %s" % op)


def get_space(tvm_op):
    _, down_graph = flatten_graph([tvm_op])
    return space.generate_space_intra_op(tvm_op, down_graph)


def cross_entorpy(label, logits):
    return torch.nn.functional.cross_entropy(logits, label)


def mse(label, logits):
    return torch.nn.functional.mse_loss(logits, label)


def calculate_loss(label_dict, logits_dict, loss_f, N=1):
    loss = 0.0
    for key, logits_lst in logits_dict.items():
        for i, logits in enumerate(logits_lst):
            label = label_dict[key][i].reshape([-1])
            logits = logits.reshape([N, -1])
            if loss_f == mse:
                label = label.expand([N, -1])
            loss += loss_f(label, logits)
    return loss


def move_data_to_device(dataset, device="cpu:0"):
    for data in dataset:
        data.graph.to(device)
        for key, lst in data.schedule_choices.items():
            tmp = []
            for val in lst:
                tmp.append(val.to(device))
            data.schedule_choices[key] = tmp
        for key, lst in data.label.items():
            tmp = []
            for val in lst:
                tmp.append(val.to(device))
            data.label[key] = tmp        


def get_decision(logits_dict):
    ret = {}
    for key, lst in logits_dict.items():
        ret[key] = []
        for val in lst:
            tmp = torch.argmax(val)
            ret[key].append(tmp)
    return ret


def count_hits(decision_dict, label_dict, ret, soft_margin=None):
    margin_lst = soft_margin if soft_margin is not None else []
    for key, lst in decision_dict.items():
        if key not in ret:
            ret[key] = [0, 0] + [0 for x in margin_lst]   # [hits, total]
        for i, val in enumerate(lst):
            resi = math.fabs(val - label_dict[key][i])
            if resi < 1e-5:
                ret[key][0] += 1
            for j, margin in enumerate(margin_lst):
                if resi < margin + 1e-5:
                    ret[key][2 + j] += 1
            ret[key][1] += 1


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """kill all child processes recursively"""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            return


class Result(object):
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def get(self, timeout=1):
        try:
            res = self.q.get(block=True, timeout=timeout)
        except Exception as e:
            res = RuntimeError(str(e))
        if self.p.is_alive():
            kill_child_processes(self.p.pid)
            self.p.terminate()
        self.p.join()
        self.q.close()
        self.q.join_thread()
        del self.p
        del self.q
        return res


def parallel_execute(op_name, shape, config, target, dev_id):
    q = multi.Queue()
    p = multi.Process(
        target=evaluate, 
        args=(op_name, shape, config, target, dev_id, q))
    p.start()
    return Result(p, q)


def run(epoch=10, pseudo_batch_size=8, train_ratio=0.8, lr=0.002, 
        check_dist=10, log_dist=100, device="cpu:0", test_only=False,
        model_file="default.pkl", train_file="train.txt", test_file="test.txt",
        continue_train=False, override_model=False, 
        use_mse=True,
        # loss_f=2,
        soft_margin=None, eval_dev=-1):
    if ((test_only or continue_train) and 
        not (os.path.exists(model_file) and os.path.isfile(model_file))):
        raise RuntimeError("Model file not found %s" % model_file)
    if ((not (test_only or continue_train)) and 
        os.path.exists(model_file) and os.path.isfile(model_file) and not override_model):
        raise RuntimeError(
            "Existing model_file %s, please remove it or set override_model=True" % model_file)
    if use_mse:
        onehot = True
        loss_f = mse
    else:
        onehot = False
        loss_f = cross_entorpy
    # if loss_f == 0:
    #     onehot = True
    #     loss_f = mse
    # elif loss_f == 1:
    #     onehot = False
    #     loss_f = cross_entorpy
    # elif loss_f == 2:
    #     # ranking loss
    #     pass

    soft_margin = [] if soft_margin is None else soft_margin
    
    if test_only:
        _train_data, _test_data = load_data(None, test_file, eval_dev)
    else:
        _train_data, _test_data = load_data(train_file, test_file, eval_dev)
    train_data = preprocess(_train_data, onehot=onehot)
    test_data = preprocess(_test_data, onehot=onehot)
    # move_data_to_device(train_data, device=device)
    # move_data_to_device(test_data, device=device)
    
    epoch = epoch
    # pseudo_batch_size = pseudo_batch_size
    pseudo_batch_size = 2

    # net = model.GNNScheduler(4, num_node_type=4, num_edge_type=5)
    net = model.PerformanceModel_ANNS()
    if test_only or continue_train:
        net.load_state_dict(torch.load(model_file))
    net.to(device)

    def _run_net(data):
        graph = data.graph
        schedule_choices = data.schedule_choices
        logits_dict = net(graph.x, graph.node_type_index, graph.edge_index, graph.edge_type_index, schedule_choices)
        return logits_dict

    def _test(full_test=True):
        logger.info("Test for accuracy and performance")
        net.eval()
        ret_dict = {}
        exceed_count = 0
        percent_lst = []
        for data in test_data:
            logits_dict = _run_net(data)
            decision_dict = get_decision(logits_dict)
            # if use_mse:
            #     label_dict = get_decision(data.label)
            # else:
            #     label_dict = data.label
                
            # if loss_f == 0:
            #     label_dict = get_decision(data.label)
            # elif loss_f == 1:
            #     label_dict = data.label
            # elif loss_f == 2:
            #     pass
            count_hits(decision_dict, label_dict, ret_dict, soft_margin=soft_margin)
            # get performance
            if _has_flextensor and full_test:
                tvm_op_lst, tvm_bufs = get_compute(data.raw.op, data.raw.shape)
                tvm_op = tvm_op_lst[-1]
                schedule_space = get_space(tvm_op)
                config = copy.deepcopy(data.raw.config)
                for key, lst in data.raw.config.op_config_lst[-1].items():
                    for i, val in enumerate(lst):
                        name = schedule_space.types[key][i]
                        subspace = schedule_space.subspaces[name]
                        index = decision_dict[key][i]
                        config.op_config_lst[-1][key][i] = subspace.get_entity(index)
                # performance of this decision
                # use process not for performance but for robustness
                dec_res = parallel_execute(data.raw.op, data.raw.shape, config, data.raw.target.target, data.raw.target.dev_id)
                dec_cost = dec_res.get(timeout=10)
                if not isinstance(dec_cost, float):
                    logger.warn("Error decision")
                    dec_cost = float("inf")
                label_res = parallel_execute(data.raw.op, data.raw.shape, data.raw.config, data.raw.target.target, data.raw.target.dev_id)
                label_cost = label_res.get(timeout=10)
                if not isinstance(label_cost, float):
                    logger.warn("Error label")
                    label_cost = float("inf")
                # s, bufs = schedule_with_config_ops(tvm_op_lst, tvm_bufs, config, target=data.raw.target.target)
                # try:
                #     dec_cost = _evaluate(s, bufs, data.raw.target.target, data.raw.target.dev_id, number=10)
                # except Exception as e:
                #     dec_cost = float("inf")
                #     logger.warn("Error decision")
                # # performance of target
                # s, bufs = schedule_with_config_ops(tvm_op_lst, tvm_bufs, data.raw.config, target=data.raw.target.target)
                # label_cost = _evaluate(s, bufs, data.raw.target.target, data.raw.target.dev_id, number=10)
                if dec_cost < label_cost:
                    exceed_count += 1
                if (dec_cost == float("inf") and label_cost == float("inf")):
                    percent_lst.append(1.0)
                else:
                    percent_lst.append(label_cost / dec_cost)
        total = len(test_data)
        logger.info("Test results:")
        string = ">>>> Accuracy: [margins: 0.0"
        num_margin = len(soft_margin)
        for margin in soft_margin:
            string += ", %.2f" % margin
        string += "]"
        logger.info(string)
        for key, lst in ret_dict.items():
            if lst[1] > 0:
                string = ">>>> ** %s ** : [%.6f" % (key, float(lst[0]) / lst[1])
                for count_margin in range(num_margin):
                    string += ", %.6f" % (float(lst[2 + count_margin]) / lst[1])
                string += "]"
                logger.info(string)
        if _has_flextensor and full_test:
            arith_mean = np.array(percent_lst).mean()
            arith_dev = np.array(percent_lst).var()
            logger.info(">>>> exceed ratio: %.6f" % (float(exceed_count) / total))
            logger.info(">>>> arithmetic average speedup: %.6f (dev %.6f)" % (arith_mean, arith_dev))
            return arith_mean
        elif _has_flextensor:
            logger.info(">>>> Performance test skipped")
            return -1
        else:
            logger.info(">>>> Performance results not availabel, consider installing flextensor")
            return -1

    if test_only:
        logger.info("Test Only.")
        _test()
        logger.info("Done!")
        return

    criterion = torch.nn.MarginRankingLoss(margin=1)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
    count_batch = 0
    print_loss = 0.0
    loss = 0.0
    best_perf = -1

    net.train()
    for ep in range(epoch):
        print_loss = 0.0
        print_loss_cnt = 0
        for shape in train_data.keys():
            torch.save(net.state_dict(), "anns_costmodel.pth")
            for ss, cost in train_data[shape]:
                count_batch += 1
                # graph = data.graph
                # schedule_choices = data.schedule_choices
                # label_dict = data.label
                # logits_dict = net(graph.x, graph.node_type_index, graph.edge_index, graph.edge_type_index, schedule_choices)
                if count_batch % 2 == 0:
                    shape1 = shape
                    ss1 = ss
                    cost1 = cost
                else:
                    shape2 = shape
                    ss2 = ss
                    cost2 = cost
                
                if count_batch == pseudo_batch_size:
                    shape1 = torch.tensor(list(shape1)).to(torch.float32)
                    shape2 = torch.tensor(list(shape2)).to(torch.float32)
                    ss1 = torch.tensor(ss1).to(torch.float32)
                    ss2 = torch.tensor(ss2).to(torch.float32)
                    cost1 = torch.tensor(cost1).to(torch.float32)
                    cost2 = torch.tensor(cost2).to(torch.float32)
                    
                    optimizer.zero_grad()
                    
                    shape1 = net.embed_shapes(shape1)
                    shape2 = net.embed_shapes(shape2)
                    predict1 = net.forward_after_query(shape1, ss1)
                    predict2 = net.forward_after_query(shape2, ss2)
                    sign = (cost1-cost2).sign().unsqueeze(0)
                    loss = criterion(predict1, predict2, sign)
                    print_loss += loss.item()
                    print_loss_cnt += 1
                    
                    loss.backward()
                    optimizer.step()
                    count_batch = 0
        
        print("Epoch=%d, Loss=%.6f" % (ep + 1, float(print_loss / print_loss_cnt)))          
        # if (ep + 1) % check_dist == 0 or ep == epoch:
        #     perf = _test()
        #     net.train()
        #     if perf - best_perf >= 0:
        #         best_perf = perf
        #         torch.save(net.state_dict(), model_file)
        #         logger.info("Model checkpoint done!")
        #     else:
        #         logger.info("Model checkpoint skipped!")
        #     if best_perf > 0:
        #         logger.info("The best performance speedup: %.6f" % best_perf)
        # if (ep + 1) % log_dist == 0:
        #     logger.info("Epoch=%d, Loss=%.6f" % (ep + 1, float(print_loss / log_dist)))
        #     print_loss = 0.0

    logger.info("Train done! Starting test...")
    # _test()
    net.eval()
    with torch.no_grad() :
        valid_loss = 0
        valid_loss_cnt = 0
        count_batch = 0
        for shape in test_data.keys():
            for ss, cost in test_data[shape]:
                count_batch += 1
                if count_batch % 2 == 0:
                    shape1 = shape
                    ss1 = ss
                    cost1 = cost
                else:
                    shape2 = shape
                    ss2 = ss
                    cost2 = cost
                
                if count_batch == pseudo_batch_size:
                    shape1 = torch.tensor(list(shape1)).to(torch.float32)
                    shape2 = torch.tensor(list(shape2)).to(torch.float32)
                    ss1 = torch.tensor(ss1).to(torch.float32)
                    ss2 = torch.tensor(ss2).to(torch.float32)
                    cost1 = torch.tensor(cost1).to(torch.float32)
                    cost2 = torch.tensor(cost2).to(torch.float32)
                    
                    shape1 = net.embed_shapes(shape1)
                    shape2 = net.embed_shapes(shape2)
                    predict1 = net.forward_after_query(shape1, ss1)
                    predict2 = net.forward_after_query(shape2, ss2)
                    sign = (cost1-cost2).sign().unsqueeze(0)
                    loss = criterion(predict1, predict2, sign)
                    valid_loss += loss.item()
                    valid_loss_cnt += 1
                    
                    count_batch = 0
    
    print ("--- Epoch {} : Train {} Valid {} ---".format(epoch, print_loss/print_loss_cnt, valid_loss/valid_loss_cnt))
    logging.info("Test done! All done!")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", help="number of epoch", type=int, default=100)
    parser.add_argument("--batch", help="pseudo batch size", type=int, default=2)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--dcheck", help="distance of checkpoint", type=int, default=10)
    parser.add_argument("--dlog", help="distance of logging", type=int, default=10)
    parser.add_argument("--dev", help="device to run on", type=str, default="cpu:0")
    parser.add_argument("--only_test", help="only test, no train", action="store_true")
    parser.add_argument("--fmodel", help="path to model file", type=str, default="default.pkl")
    parser.add_argument("--ftrain", help="path to train data file", type=str, default="train_anns.txt")
    parser.add_argument("--ftest", help="path to test data file", type=str, default="test_anns.txt")
    parser.add_argument("--retrain", help="continue to train existing model", action="store_true")
    parser.add_argument("--override", help="override existing model", action="store_true")
    parser.add_argument("--use_mse", help="use mse loss, otherwise cross_entorpy", action="store_true")
    # parser.add_argument("--loss_f", help="0: mse loss, 1: cross_entorpy, 2: ranking loss", type=int, default=2)
    parser.add_argument("--eval_dev", help="device id to evaluate performance", type=int, default=-1)
    parser.add_argument("--flog", help="filename of log", type=str, default="log_train.txt")

    args = parser.parse_args()

    formatter = logging.Formatter("[%(levelname)s] %(message)s (%(asctime)s)")

    fh = logging.FileHandler(args.flog)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("############# New Task ###########")

    margin = [1, 2, 3, 4, 5]
    run(
        epoch=args.epoch,
        pseudo_batch_size=args.batch,
        lr=args.lr,
        check_dist=args.dcheck,
        log_dist=args.dlog,
        device=args.dev,
        test_only=args.only_test,
        model_file=args.fmodel,
        train_file=args.ftrain,
        test_file=args.ftest,
        continue_train=args.retrain,
        override_model=args.override,
        use_mse=args.use_mse,
        # loss_f=args.loss_f,
        soft_margin=margin,
        eval_dev=args.eval_dev
    )