#%%
import pandas as pd
import matplotlib.pyplot as plt
# %%
onion = pd.read_csv("stress relaxation.csv")
# %%
onion.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# %%
onion
# %%
fig, ax = plt.subplots(1,1, figsize=(10,7))
ax.plot(onion[0], onion[1])
ax.plot(onion[3], onion[4])
ax.plot(onion[6] ,onion[7])
# %%
