import plotly.plotly as py
import numpy as np
import plotly.tools as tls
from plotly.graph_objs import *
py.sign_in('cwschupp', 'ul2e0se0og')


def bsa_linechart(dic, ordered_trt_list, cluster):
    '''
    INPUT: for a given cluster, takes in the dictionary of linear model
           coefficients and for each of the most common treatments
           plots the model
    OUTPUT: url for the plotly figure
    '''
    trace1 = Scatter(
        x=range(1, 37),
        y=[(dic[ordered_trt_list[0]]['bsa'][1]+dic[ordered_trt_list[0]]['bsa'][0]*t) 
           for t in range(1, 37)], 
        name=ordered_trt_list[0]
        )
    trace2 = Scatter(
        x=range(1, 37),
        y=[(dic[ordered_trt_list[1]]['bsa'][1]+dic[ordered_trt_list[1]]['bsa'][0]*t) 
           for t in range(1, 37)], 
        name=ordered_trt_list[1]
        )
    trace3 = Scatter(
        x=range(1, 37),
        y=[(dic[ordered_trt_list[2]]['bsa'][1]+dic[ordered_trt_list[2]]['bsa'][0]*t) 
           for t in range(1, 37)], 
        name=ordered_trt_list[2]
        )
            
    data = Data([trace1, trace2, trace3])

    layout = Layout(
                    barmode='group',
                    title='Psoriasis Improvement in Body Surface Affected',
                    font=Font(size=20),
                    showlegend=True,
                    paper_bgcolor='rgb(240, 240, 240)',
                    plot_bgcolor='rgb(240, 240, 240)',
                    legend=Legend(
                        font=Font(size=16),
                        x=0.1
                        ),
                    yaxis=YAxis(
                        title='Percent Reduction in Body Surface Affected',
                        font=Font(size=18),
                        range=[0, 30],
                        showgrid=True,
                        gridcolor='rgb(223, 223, 223)',
                        gridwidth=1
                            ),
                    xaxis=XAxis(
                        title='Time (Months)',
                        font=Font(size=18),
                        gridcolor='rgb(223, 223, 223)',
                        gridwidth=1
                        )
        )
    fig = Figure(data=data, layout=layout)
    filename = 'cluster_'+str(cluster)+'_bsa_linechart'
    
    return py.plot(fig, filename=filename, fileopt='new', auto_open=False)

if __name__ == '__main__':
    d = {'biologic': {'bsa': np.array([0.58770119, -5.28744242]),
          'pga': {'<=12': 0.0075949367088607592,
           '<=24': 0.33417721518987342,
           '<=36': 0.67341772151898738}},
         'biologic, phototherapy': {'bsa': np.array([0.34520405, -1.63439082]),
          'pga': {'<=12': 0.028806584362139918,
           '<=24': 0.41975308641975306,
           '<=36': 0.75720164609053497}},
         'phototherapy': {'bsa': np.array([0.66933966, -6.70842942]),
          'pga': {'<=12': 0.0092165898617511521,
           '<=24': 0.33179723502304148,
           '<=36': 0.72811059907834097}},
         'topical': {'bsa': np.array([0.91592764, -9.61312416]),
          'pga': {'<=12': 0.0086834733893557427,
           '<=24': 0.29495798319327732,
           '<=36': 0.69159663865546217}},
         'topical, biologic': {'bsa': np.array([0.67216044, -6.73241649]),
          'pga': {'<=12': 0.009442870632672332,
           '<=24': 0.30500472143531632,
           '<=36': 0.69971671388101986}},
         'topical, phototherapy': {'bsa': np.array([0.70984317, -7.20284488]),
          'pga': {'<=12': 0.0099099099099099093,
           '<=24': 0.33603603603603605,
           '<=36': 0.70990990990990988}}}
    trt_list= ['biologic', 'topical', 'phototherapy']

    print bsa_linechart(d, trt_list, 1)
