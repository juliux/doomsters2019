#!/usr/bin/env python

# +-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+
# |V|I|Z|D|O|O|M| |B|O|T| |2|0|1|9|
# +-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+

from termcolor import colored

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# - Banner generation


# +-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+
# |C|L|A|S|S|E|S| |D|E|F|I|N|I|T|I|O|N|
# +-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


# +-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+
# |D|E|P|L|O|Y|M|E|N|T| |L|O|G|I|C|
# +-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+

# - Setup logger

botLogger = logging.getLogger(__name__)
f_handler = logging.FileHandler('sample_report.log')
f_handler.setLevel(logging.ERROR)
f_format = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
f_handler.setFormatter(f_format)
botLogger.addHandler(f_handler)

# - Main banner printing

