import numpy as np
import matplotlib.pyplot as plt
from pyxu.util.misc import star_like_sample

star = star_like_sample(N=256, w=8, s=20, po=3, x0=0.7)
plt.figure()
plt.imshow(star)