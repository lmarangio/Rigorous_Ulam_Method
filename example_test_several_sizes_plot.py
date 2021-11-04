import pickle
from sage.all import *

with open('test_several_sizes.pickle', 'rb') as f:
	results = pickle.load(f)

results = [x for x in results if x['n'] >= 1024]

ns = [x['n'] for x in results]
assemble_times = [x['assemble_time'] for x in results]
Ci_times = [x['Ci_time'] for x in results]
residual_times = [x['residual_time'] for x in results]

p = list_plot([(x['n'], x['error']) for x in results], title='Decay of the error for 1-grid estimation', scale='loglog', plotjoined=True, legend_label='error', marker='+')
p.show()

def plot_decay(x, color):
	return list_plot(x['Ci'], plotjoined=True, title='Decay of Ci vs. grid size', scale='semilogy', legend_label='%s' % x['n'], color=color)

p = sum(plot_decay(x, hue(float(i)/len(results))) for i, x in enumerate(results))
p.show()

with open('test_several_sizes_twogrid_results.pickle', 'rb') as f:
	twogrid_results = pickle.load(f)

points = [(time, error) for (time, error, label) in twogrid_results]

p = scatter_plot(points, scale='loglog')
for time, error, label in twogrid_results:
	p = p + text(label, (time*1.1, error*1.1))

p.show()