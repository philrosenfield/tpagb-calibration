from ..fileio import replace_ext, get_files



# I did this in ipython
# triouts = !! ls out*dat
# [trilegal.utils.trilegal2matchphot(t, extra='_cor') for t in triouts]
# then copied fake and param files to the same directory and uploaded to odyssey.
# targets = np.unique([t.split('_')[1] for t in triouts])

i = 1  # command counter
k = 0  # job array script counter
nproc = 32  # how mand commands per job array (itc_cluster = 64, conroy = 32, imac ~10)
slurm = True # write scripts for job array [True] write one big script with waits [False]

# where to find calcsfh binary
calcsfh = '/n/home01/prosenfield/match2.5/bin/calcsfh'

# calcsfh flags
flags = '-PARSEC -mcdata -kroupa -zinc'

# jobarray script format
sfmt = 'calcsfh_trilegal_script_{}.sh'

header = 'calcsfh="{}"\n'.format(calcsfh)
line = header

phots = [[o for o in triouts if t in o] for t in targets]

ntot = len(triouts)

for j, t in enumerate(targets):
    param  = get_files('.', '*{}*param'.format(t))[0][2:]
    fake  = get_files('.', '*{}*fake'.format(t))[0][2:]
    for phot in phots[j]:
        out = replace_ext(phot, '.out')
        scrn = replace_ext(phot, '.scrn')
        line += '$calcsfh {0} {1} {2} {3} {4} > {5}\n'.format(param, phot, fake, out, flags, scrn)
        if i % nproc == 0 or i == ntot:
            if slurm:
                # dump the commands to a file to be called by the job array
                with open(sfmt.format(k), 'w') as outp:
                    outp.write(line)
                # start over
                line = header
                k += 1
            else:
                # just put in a wait for the rest of the jobs to complete
                l += 'wait\n'
        i += 1

if not slurm:
    with open(smft.format(k), 'w') as outp:
        outp.write(line)
