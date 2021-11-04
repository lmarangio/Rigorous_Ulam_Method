
from noise_induced_order_azazello import *
from noise_make_graph import *

create_graph_and_sheet(D,
                       [p[1:] for p in params],
                       scale = "semilogx",
                       xmin = 0.01, xmax = 0.1,
                       ymin = -0.5, ymax = 0.8) #plot_args        
