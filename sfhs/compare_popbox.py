import os
import sys
from .popbox import PopBox, compare_popboxes

dn = '/Users/rosenfield/Desktop/Leo/noAGB'
dw = '/Users/rosenfield/Desktop/Leo/withAGB'

pboxnoagbs = [os.path.join(dn, l) for l in os.listdir(dn)
              if l.endswith('popbox')]

pboxwithagbs = [os.path.join(dw, l) for l in os.listdir(dw)
                if l.endswith('popbox')]

pbsnoagb = [PopBox(p) for p in pboxnoagbs]
pbswithagb = [PopBox(p) for p in pboxwithagbs]

assert len(pbsnoagb) == len(pbsnoagb), 'Mismatch!'

for i in range(len(pbsnoagb)):
    reg = pbsnoagb[i].name.split('.')[0].replace('region_', '').replace('_','-')
    titles =['{0:s} with AGB'.format(reg),
             '{0:s} without AGB'.format(reg),
             'with - without']
    outfig = 'compare_popbox_{0:s}.png'.format(reg)
    compare_popboxes(pbswithagb[i], pbsnoagb[i], titles=titles, outfig=outfig)
