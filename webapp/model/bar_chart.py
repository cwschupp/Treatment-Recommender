import plotly.plotly as py
import plotly.tools as tls  
from plotly.graph_objs import *
from collections import Counter
py.sign_in('cwschupp', 'ul2e0se0og')


def barchart(dict, num_bars, cluster):
    '''
    INPUT: for a given cluster, takes in the counter of treatment
           with number of patients receiving 
    OUTPUT: url for the plotly figure
    '''
    trace1 = Bar(
        x=[tup[0] for tup in dict.most_common(num_bars)],
        y=[tup[1]/float(sum(dict.values())) for tup in dict.most_common(num_bars)]   
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
    filename = 'cluster_'+str(cluster)+'_barchart'
    
    return py.plot(fig, filename=filename, fileopt='new', auto_open=False)

if __name__ == '__main__':
    c = Counter({'topical, biologic': 1544, 
                 'topical, phototherapy': 1285, 
                 'topical': 879, 
                 'biologic, phototherapy': 745, 
                 'phototherapy': 541, 
                 'biologic': 511, 
                 'topical, biologic, phototherapy': 266})
    print barchart(c, 4, 1)
















