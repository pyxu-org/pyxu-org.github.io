import numpy as np
import pyxu.experimental.xray as pxr

op = pxr.XRayTransform.init(
    arg_shape=(5, 6),
    origin=0,
    pitch=1,
    method="ray-trace",
    n_spec=np.array([[1   , 0   ],  # 3 rays ...
                     [0.5 , 0.5 ],
                     [0.75, 0.25]]),
    t_spec=np.array([[2.5, 3],  # ... all defined w.r.t volume center
                     [2.5, 3],
                     [2.5, 3]]),
)
fig = op.diagnostic_plot()
fig.show()