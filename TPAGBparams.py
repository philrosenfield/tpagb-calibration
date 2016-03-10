import os
import socket
hostname = socket.gethostname()

#EXT = '.png'
EXT = '.pdf'
if hostname.endswith('astro.washington.edu'):
    print 'better edit TPAGBparams.py...'
else:
    if 'Linux' in os.uname():
        # linux laptop
        research_path = '/home/phil/research/'
        if os.uname()[1] == 'andromeda':
            # unipd
            research_path = '/home/rosenfield/research/'
            import matplotlib as mpl
            mpl.use('Agg')
    else:
        # mac
        #research_path = '/Users/rosenfield/research/'
        research_path = '/Volumes/tehom/research'
tpcalib_dir = os.path.join(research_path, 'TP-AGBcalib')
snap_src = os.path.join(tpcalib_dir, 'SNAP')
phat_src = os.path.join(tpcalib_dir, 'PHAT')

matchfake_loc = os.path.join(snap_src, 'data', 'galaxies')
data_loc = os.path.join(snap_src, 'data', 'opt_ir_matched_v2')

dirs = [tpcalib_dir, snap_src, phat_src]
for d in dirs:
    if not os.path.isdir(d):
        #print 'Warning. Can not find %s.' % d
        pass
