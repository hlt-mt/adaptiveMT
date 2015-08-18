#!/usr/bin/python

import sys, getopt


if __name__ == "__main__":
        try:
                opts, args = getopt.getopt(sys.argv[1:], "l:b:t:", ["log=", "baseline=", "tm-only="])
        except getopt.GetoptError:
                sys.exit()

        for opt, arg in opts:
                if opt in ("-l", "--log"):
                        log = open(arg)
                elif opt in ("-b", "--baseline"):
                        bsl = [line.strip() for line in open(arg)]
		elif opt in ("-t", "--tm-only"):
			tmo = [line.strip() for line in open(arg)]
		else:
			pass
	
	log_out = log.readline().strip()
	while log_out:
		parts = log_out.split(':')
		if len (parts) == 3:
			num_s = int(parts[2])
			print num_s, ":"
			baseline = bsl[num_s-1].strip()
			tm_only = tmo[num_s-1].strip()
			inp = log.readline().strip().split(": ")[1].strip()
			out = log.readline().strip().split(": ")[1].strip()
			reference = log.readline().strip().split(": ")[1].strip()
			print "BSL:", baseline
			print "TMO:", tm_only
			print "OUT:", out
			print "REF:", reference
			print "INP:", inp
			print 20*'*'
		log_out = log.readline().strip()
	
