import pandas as pd
import numpy as np
# export IFM data
def exportIFM(out, saveIFM, firstBatch, fname):
    if saveIFM and firstBatch:
        d = out.detach().cpu().numpy()[0]
        print(fname)
        print(np.shape(d))
        d = np.reshape(out.detach().cpu().numpy()[0], -1, 'C')
        d[d > 1e-6] = 1
        d[d < -1e-6] = 1
        d[d != 1] = 0
        d = d.astype(int)
        print(np.shape(d))
        pd.DataFrame(d).to_csv(fname)
