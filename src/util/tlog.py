from termcolor import colored

def tlog(s, family='debug'):
	'''
	prints a string with an appropriate header
	'''
	if family == 'note':
		header = colored('[Note]','red','on_cyan')
	elif family == 'iter':
		header = colored('    ','grey','on_white')
	elif family == 'debug':
		header = colored('[Debug]','grey','on_yellow')
	elif family == 'error':
		header = colored('[Error]','cyan','on_red')
	else:
		header = colored('[Invalid print family]','white','on_red')

	out = '{} {}'.format(header, s)
	print(out)
