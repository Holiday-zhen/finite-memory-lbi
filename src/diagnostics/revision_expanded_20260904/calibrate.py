import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'engine'))
import experiments_adaptive_revision as ad
from threadpoolctl import threadpool_limits
if __name__=='__main__':
    start=time.perf_counter()
    with threadpool_limits(1):
        selection=ad.experiment_1_calibration()
    print('COMPLETE',selection,'seconds',time.perf_counter()-start,flush=True)
