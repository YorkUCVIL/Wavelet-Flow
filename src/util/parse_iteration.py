def parse_iteration(s):
	if s[-1] == 'k':
		return int(s[:-1])*1000
	elif s[-1] == 'm':
		return int(s[:-1])*1000000
	else:
		return int(s)
