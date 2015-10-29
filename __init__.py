import matplotlib.pyplot as plt
try:
   plt.style.use('presentation')
except:
   print(mpl.get_configdir())
   print(plt.style.available)
from .fileio import *

