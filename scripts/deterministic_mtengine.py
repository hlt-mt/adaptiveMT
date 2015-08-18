#!/usr/bin/python

import sys, logging, os
import codecs, subprocess, select, re, logging
from decoder import Decoder_Deterministic 

from ConfigParser import SafeConfigParser

import argparse

logging.basicConfig(level=logging.INFO)

def usage():
	"""
	Prints script usage.
	"""
	sys.stderr.write("./deterministic_mtengine.py -source srcfile -target trgfile\n")

if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('-source', dest="srcfile", help='source text', default="")
	parser.add_argument('-target', dest="trgfile", help='target text', default="")
	parser.add_argument('-print-id', action="store_true", help='flag for printing segment id')
	parser.add_argument('-report-segmentation', action="store_true", help='flag for printing phrase segmentation')
	parser.add_argument('-print-passthrough', action="store_true", help='flag for printing passthrough tag')
	parser.add_argument('-verbosity', help='verbosity level, default: 0', type=int, default=0)

        args = parser.parse_args(sys.argv[1:])

	if args.srcfile == "" or args.trgfile == "":
		usage()

        # create configfile for Deterministic decoder
        decoder_config = SafeConfigParser()

	decoder_config.add_section('decoder')
	decoder_config.set('decoder', 'source', args.srcfile)
	decoder_config.set('decoder', 'target', args.trgfile)
	decoder_config.set('decoder', 'verbosity', repr(args.verbosity))

	Decoder_object = Decoder_Deterministic(decoder_config)
	
	# main loop
	# initialize: first sentence has no history
	source = sys.stdin.readline().strip()
	s_id = 1
	segid = None
	passthrough = None

	re_passthrough = re.compile(r"(<passthrough[^>]*>)")
	re_segid = re.compile(r"<seg[^>]*id=\"(.+)\"[^>]*>(.+)</seg>")
	while source:
		logging.info(str(s_id))

		tmp = re_segid.match (source)
                if tmp != None:
			segid = tmp.group(1)
			source = tmp.group(2)	

		tmp = re_passthrough.search (source)
                if tmp != None:
                        passthrough = tmp.group(1)
			source = re.sub (re_passthrough, '', source)

		# talk to decoder
		decoder_out, decoder_err = Decoder_object.communicate(source)
                logging.info("DECODER_OUT: "+decoder_out)
                logging.info("DECODER_ERR: "+decoder_err)
		
		if args.print_id:
	                if segid != None: 
                		sys.stdout.write(segid+' ')

                if args.print_passthrough:
                        if passthrough != None:
                		sys.stdout.write(passthrough+'')

		if args.report_segmentation:
			if decoder_out != '':
				source_len = len(source.split())
				phrase_align_out = "|0-" + repr((source_len-1)) + "|"
	                	decoder_out += ' ' + phrase_align_out

                # write translation to stdout
                sys.stdout.write(decoder_out+'\n')
                sys.stdout.flush()
	
		source = sys.stdin.readline().strip()

		s_id += 1

