import numpy as np
import torch
from model import PerformanceModel_ANNS
import hnswlib
import time
from train_anns import load_data, preprocess

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _train_data, _ = load_data("gemm_1207_total.log", None, -1)
    train_data = preprocess(_train_data)

    net = PerformanceModel_ANNS()
    net = net.to(device)
    net.load_state_dict(torch.load('anns_costmodel.pth'))
    net.eval()
    
    start = time.time()  
    embeddings = [] 
    
    for shape in train_data.keys():
        for ss, cost in train_data[shape]:
            ss = torch.tensor(ss).to(torch.float32)
            cost = torch.tensor(cost).to(torch.float32)
            ss = ss.to(device)
            embedding = net.embed_parameters(ss)
            embeddings.append(embedding.detach().cpu().tolist())
          
    embeddings = np.array(embeddings)
    print("Calculate Embedding : ", time.time()-start)

    print(embeddings.shape)
    dim = embeddings.shape[1]
    num_elements = embeddings.shape[0]
    p = hnswlib.Index(space = 'l2', dim = dim) 
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 32)
    start = time.time()
    p.add_items(embeddings, np.arange(num_elements))
    print("Gen Index : ", time.time()-start)
    p.save_index("hnsw_schedule.bin")