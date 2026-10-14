import itertools
import numpy as np
import pandas as pd
from src.analysis.rotation_measurement import rotation_measurement_bounds, float32_rounding_cell


def test_bounds_enclose_memberwise_price_scenarios_and_keep_one_way_semantics():
    for rows in [
        [[10.,9.,11.,.4],[10.,11.,9.,.6]],
        [[10.,10.,10.,.4],[10.,10.001,9.999,.6]],
        [[10.,9.,11.,.4],[10.,9.,11.,.6]],
    ]:
        p=pd.DataFrame(rows,columns=['open','close_0945','close_1030','weight'])
        p.iloc[:,:3]=p.iloc[:,:3].astype('float32').astype('float64')
        base=rotation_measurement_bounds(p,opening_halfwidth=0.)
        stress=rotation_measurement_bounds(p,opening_halfwidth=.005)
        assert stress['lower']<=base['lower'] and stress['upper']>=base['upper']
        choices=[]
        for row in p.itertuples(index=False):
            lo,hi=float32_rounding_cell(np.array(row[:3]))
            lo[0]-=.005;hi[0]+=.005
            choices.append(list(itertools.product(*[(lo[i],row[i],hi[i]) for i in range(3)])))
        for state in itertools.product(*choices):
            up=sum(w for (o,a,b),w in zip(state,p.weight) if a<o<b)
            down=sum(w for (o,a,b),w in zip(state,p.weight) if b<o<a)
            value=2*min(up,down)
            assert stress['lower']-1e-12<=value<=stress['upper']+1e-12
    assert stress['point']==stress['lower']==stress['upper']==0.
