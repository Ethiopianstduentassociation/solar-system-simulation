import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="9-Planet Solar System Simulation", layout="centered")
st.title("9-Planet Solar System N-Body Simulation")
st.write(
    "Modeling Newtonian gravity and Keplerian orbital stability using Velocity"
    " Verlet integration."
)

# Interactive slider for college admissions officers to test
years = st.slider(
    "Simulation Duration (Earth Years)", min_value=1, max_value=250, value=50
)

G = 4 * np.pi**2
bodies_data = [
    ["Sun", 1.0, 0.0, 0.0, "yellow", 12],
    ["Mercury", 1.65e-7, 0.39, np.sqrt(G / 0.39), "gray", 3],
    ["Venus", 2.45e-6, 0.72, np.sqrt(G / 0.72), "orange", 5],
    ["Earth", 3.00e-6, 1.00, np.sqrt(G / 1.00), "deepskyblue", 5],
    ["Mars", 3.21e-7, 1.52, np.sqrt(G / 1.52), "red", 4],
    ["Jupiter", 9.54e-4, 5.20, np.sqrt(G / 5.20), "sandybrown", 9],
    ["Saturn", 2.86e-4, 9.58, np.sqrt(G / 9.58), "gold", 8],
    ["Uranus", 4.37e-5, 19.22, np.sqrt(G / 19.22), "lightblue", 6],
    ["Neptune", 5.15e-5, 30.05, np.sqrt(G / 30.05), "royalblue", 6],
    ["Pluto", 6.55e-9, 39.48, np.sqrt(G / 39.48), "lightgray", 3],
]

n_bodies = len(bodies_data)
names = [b for b in bodies_data]
masses = np.array([b for b in bodies_data])
colors = [b for b in bodies_data]
sizes = [b for b in bodies_data]

pos = np.zeros((n_bodies, 2))
vel = np.zeros((n_bodies, 2))
for i in range(n_bodies):
  pos[i, 0] = bodies_data[i]
  vel[i, 1] = bodies_data[i]

vel = -np.sum(masses[1:, None] * vel[1:], axis=0) / masses


def compute_accelerations(positions):
  acc = np.zeros_like(positions)
  for i in range(n_bodies):
    for j in range(n_bodies):
      if i != j:
        r_vec = positions[j] - positions[i]
        r_mag = np.linalg.norm(r_vec)
        acc[i] += G * masses[j] * r_vec / (r_mag**3)
  return acc


dt = 0.02
steps = int(years / dt)
acc = compute_accelerations(pos)
history = [[] for _ in range(n_bodies)]

for step in range(steps):
  pos += vel * dt + 0.5 * acc * (dt**2)
  new_acc = compute_accelerations(pos)
  vel += 0.5 * (acc + new_acc) * dt
  acc = new_acc
  if step % 5 == 0:  # Save every 5th point for smooth plotting
    for i in range(n_bodies):
      history[i].append(pos[i].copy())

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(-42, 42)
ax.set_ylim(-42, 42)
ax.set_xlabel("Distance (AU)", color="white")
ax.set_ylabel("Distance (AU)", color="white")
ax.tick_params(colors="white")

for i in range(n_bodies):
  trail = np.array(history[i])
  if len(trail) > 0:
    ax.plot(
        trail[:, 0], trail[:, 1], "-", color=colors[i], alpha=0.5, linewidth=1
    )
  ax.plot(
      pos[i, 0],
      pos[i, 1],
      "o",
      color=colors[i],
      markersize=sizes[i],
      label=names[i],
  )

ax.legend(
    loc="upper right",
    facecolor="black",
    edgecolor="white",
    labelcolor="white",
    fontsize="x-small",
)
st.pyplot(fig)
