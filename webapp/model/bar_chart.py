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
    /float(sum(c.values()))
    '''
    trace1 = Bar(
        x=[tup[0] for tup in dict.most_common(num_bars)],
        y=[tup[1] for tup in dict.most_common(num_bars)],
        type='percent'   
        )
    
    title = 'The Top Treatments Currently Used'

    layout = Layout(
        title = title,
        showlegend=False,
        yaxis = YAxis(
            title = 'Percent Patients Receiving Treatment',
            gridcolor='white'
            ),
        xaxis = XAxis(
            title = 'Treatments')
        )
            
    data = Data([trace1])
    fig = Figure(data=data, layout=layout)
    filename = 'cluster_'+str(cluster)+'_barchart'
    
    return py.plot(data, filename=filename, fileopt='new', auto_open=False)

if __name__ == '__main__':
    c = Counter({'topical, biologic': 1544, 
                 'topical, phototherapy': 1285, 
                 'topical': 879, 
                 'biologic, phototherapy': 745, 
                 'phototherapy': 541, 
                 'biologic': 511, 
                 'topical, biologic, phototherapy': 266})
    print barchart(c, 4, 1)
















