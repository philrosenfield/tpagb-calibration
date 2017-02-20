import matplotlib.pyplot as plt
from astropy.io import fits

x1 = 3.
x2 = 5.
y1 = 23.6
y2 = 26.3
m = (y2-y1) / (x2 - x1)
b = 19.40
x = np.linspace(3, 6.5)
y = m * x + b

data = fits.getdata('hlsp_phat_hst_wfc3-uvis-acs-wfc-wfc3-ir_12070-m31-b23_f275w-f336w-f475w-f814w-f110w-f160w_v2_st.fits')

fig, ax = plt.subplots()
mag = data['F475W_VEGA']
mag2 = data['F814W_VEGA']
color = mag - mag2
ax.plot(color, mag, '.', ms=3, alpha=0.3)

print(i+1)
for i in range(len(x)-1):
    v = np.array([[x[i], y[i]], [x[i], 20], [x[i+1],  20], [x[i+1], y[i]], [x[i], y[i]]])
    ax.plot(v[:, 0], v[:,1])
    print('{0:.4f} {1:.4f} {0:.4f} {2:.4f} {3:.4f} {2:.4f} {3:.4f} {1:.4f}'.format(x[i], y[i], 20, x[i+1]))

ax.invert_yaxis()
ax.set_xlim(1, 6.5)
ax.set_ylim(28, 20)
