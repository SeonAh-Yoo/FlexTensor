import numpy as np
import torch
from flextensor.model import PerformanceModel_ANNS
import hnswlib
from flextensor.train_anns import load_data, preprocess

def _anns_schedule(shape_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _train_data, _ = load_data("gemm_1207_total.log", None, -1)
    train_data = preprocess(_train_data)

    net = PerformanceModel_ANNS()
    net = net.to(device)
    net.load_state_dict(torch.load('anns_costmodel.pth'))
    net.eval()
    
    schedules = []
    for shape in train_data.keys():
        for ss, cost in train_data[shape]:
            ss = torch.tensor(ss).to(torch.float32)
            cost = torch.tensor(cost).to(torch.float32)
            ss = ss.to(device)
            schedules.append(ss)

    dim = 128
    num_elements = schedules.__len__()

    p = hnswlib.Index(space='ip', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
    p.load_index("hnsw_schedule.bin", max_elements = num_elements)
    p.set_ef(200) # ef should always be > k

    shape = torch.tensor(list(shape_)).to(torch.float32)
    shape = shape.to(device)
    query = net.embed_shapes(shape)
    labels, distances = p.knn_query(query.cpu().detach().numpy(), k=20)
    
    schedules = [ss.cpu() for ss in schedules]
    
    return list(np.array(schedules)[labels[0]])