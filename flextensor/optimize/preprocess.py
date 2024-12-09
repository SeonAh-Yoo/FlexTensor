import numpy as np

ratio = 0.8

if __name__ == "__main__":
    raw = []
    
    with open("gemm_1207_total.log", "r") as fin:
        for line in fin:
            raw.append(line)
    if not raw[-1][-1] == "\n":
        raw[-1] = raw[-1] + "\n"
        
    shapes = [(2**i, 2**j, 2**k) for i in range(5, 11) for j in range(5, 11) for k in range(5, 11)]         
    np.random.shuffle(shapes)
    length = 173
    train = shapes[:length]
    test = shapes[length:]
    with open("train_anns.txt", "w") as fout:
        for s in train:
            for line in raw:
                if str(s) in line:
                    fout.write(line)
    with open("test_anns.txt", "w") as fout:
        for s in test:
            for line in raw:
                if str(s) in line:
                    fout.write(line)


    # for i in range(8):
    #     with open("gemm_1203_{0}.log".format(i), "r") as f:
    #         for line in f:
    #             raw.append(line)
    # raw.sort()
    # with open("gemm_1207_total.log", "w") as f:
    #     f.writelines(raw)