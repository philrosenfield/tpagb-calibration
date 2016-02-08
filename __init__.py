import seaborn as sns
sns.set()
sns.set_context('paper')
import matplotlib.pyplot as plt
try:
   plt.style.use('paper')
except:
   print(mpl.get_configdir())
   print(plt.style.available)
from .fileio import *
