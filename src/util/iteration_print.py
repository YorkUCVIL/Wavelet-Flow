from util import *
import numpy as np

def iteration_print(header, values):
	out_str = str(header) + '|'
	for kv in values:
		out_str += ' {}: '.format(colored(kv[0], 'white', attrs=['underline']))
		if type(kv[1]) == float or type(kv[1]) == np.float32:
			out_str += '{:.4f}'.format(kv[1])
		else:
			out_str += '{}'.format(kv[1])
	tlog(out_str,'iter')
