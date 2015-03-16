import plotly.plotly as py
import numpy as np
import plotly.tools as tls  
from plotly.graph_objs import *
from collections import Counter
py.sign_in('cwschupp', 'ul2e0se0og')

def treatment_barchart(dic, ordered_trt_list, cluster):
    '''
    INPUT: for a given cluster, takes in the counter of treatment
           with number of patients receiving, the N most common Treatments
    OUTPUT: url for the plotly figure
    '''
    trace1 = Bar(
        x=ordered_trt_list,
        y=[dic[trt]/float(sum(dic.values())) for trt in ordered_trt_list]   
        )
            
    data = Data([trace1])

    layout = Layout(
                    title = 'The Top Treatments Currently Used',
                    showlegend=False,
                    yaxis = YAxis(
                        title = 'Percent Patients Receiving Treatment',
                        gridcolor='white'
                            ),
                    xaxis = XAxis(
                        title = 'Treatments')
        )

    fig = Figure(data=data, layout=layout)
    filename = 'cluster_'+str(cluster)+'_trt_barchart'
    
    return py.plot(fig, filename=filename, fileopt='new', auto_open=False)

def pga_barchart(dic, ordered_trt_list, time_list, cluster):
    '''
    INPUT: for a given cluster, takes in the relative frequency distribution
           of change in PGA over time for each treatment
    OUTPUT: url for the plotly figure
    '''
    trace1 = Bar(
        x=time_list,
        y=[dic[ordered_trt_list[0]]['pga'][t] for t in time_list], 
        name=ordered_trt_list[0]
        )
    trace2 = Bar(
        x=time_list,
        y=[dic[ordered_trt_list[1]]['pga'][t] for t in time_list], 
        name=ordered_trt_list[1]
        )
    trace3 = Bar(
        x=time_list,
        y=[dic[ordered_trt_list[2]]['pga'][t] for t in time_list], 
        name=ordered_trt_list[2]
        )
            
    data = Data([trace1, trace2, trace3])

    layout = Layout(
                    barmode='group',
                    title = 'Psoriasis Severity Improvement',
                    showlegend=True,
                    legend = Legend(
                        x=0
                        ),
                    yaxis = YAxis(
                        title = 'Percent Patients That Improved 1 Level or More',
                        gridcolor='white'
                            ),
                    xaxis = XAxis(
                        title = 'Time (Months)')
        )
    fig = Figure(data=data, layout=layout)
    filename = 'cluster_'+str(cluster)+'_pga_barchart'
    
    return py.plot(fig, filename=filename, fileopt='new', auto_open=False)

if __name__ == '__main__':
    c = Counter({'topical, biologic': 1544, 
                 'topical, phototherapy': 1285, 
                 'topical': 879, 
                 'biologic, phototherapy': 745, 
                 'phototherapy': 541, 
                 'biologic': 511, 
                 'topical, biologic, phototherapy': 266})
    trt_list= ['biologic','topical', 'phototherapy']
    print treatment_barchart(c, trt_list, 1)

    d= {'biologic': {'bsa': np.array([ 0.58770119, -5.28744242]),
          'pga': {'<=12': 0.0075949367088607592,
           '<=24': 0.33417721518987342,
           '<=36': 0.67341772151898738}},
         'biologic, phototherapy': {'bsa': np.array([ 0.34520405, -1.63439082]),
          'pga': {'<=12': 0.028806584362139918,
           '<=24': 0.41975308641975306,
           '<=36': 0.75720164609053497}},
         'phototherapy': {'bsa': np.array([ 0.66933966, -6.70842942]),
          'pga': {'<=12': 0.0092165898617511521,
           '<=24': 0.33179723502304148,
           '<=36': 0.72811059907834097}},
         'topical': {'bsa': np.array([ 0.91592764, -9.61312416]),
          'pga': {'<=12': 0.0086834733893557427,
           '<=24': 0.29495798319327732,
           '<=36': 0.69159663865546217}},
         'topical, biologic': {'bsa': np.array([ 0.67216044, -6.73241649]),
          'pga': {'<=12': 0.009442870632672332,
           '<=24': 0.30500472143531632,
           '<=36': 0.69971671388101986}},
         'topical, phototherapy': {'bsa': np.array([ 0.70984317, -7.20284488]),
          'pga': {'<=12': 0.0099099099099099093,
           '<=24': 0.33603603603603605,
           '<=36': 0.70990990990990988}}}
    trt_list= ['biologic','topical', 'phototherapy']
    time_list = ['<=12', '<=24', '<=36']

    print pga_barchart(d, trt_list, time_list, 1)
