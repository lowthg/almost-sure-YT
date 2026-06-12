import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.fft import fft, ifft, fftfreq

# Constants
hbar = 1  # Reduced Planck's constant
m = 1     # Mass of the particle
k = 1     # Strength of the potential
n = 600   # Grid size
L = 20    # Length of the domain
dx = 2 * L / n  # Spatial resolution
dt = 0.02  # Time step
timesteps = 10000  # Number of time steps

# Spatial grid
x = np.linspace(-L, L, n)

# Potential V(x) = k * x^4
V = k * x**4
# V = (x/30)**4 * 40
# V = (x/30)**2 * 40
# V = x*0

# Initial wavefunction: Gaussian centered at x0
x0 = -10  # Initial position
sigma = np.sqrt(0.5)  # Width of Gaussian
sigma=1
psi0 = (1/((2 * np.pi * sigma**2)**0.25)) * np.exp(-(x - x0)**2 / (4 * sigma**2))

# Normalize the wavefunction
psi0 /= np.linalg.norm(psi0)
# psi0 = psi0 * np.exp(2j * x)

# Kinetic energy operator in momentum space
def kinetic_operator():
    k_space = fftfreq(n, dx) * (2 * np.pi)  # Momentum space grid
    T = -0.5 * (hbar**2 / m) * (1j * k_space)**2  # Kinetic energy operator
    return T

# Time evolution using the full Hamiltonian
def time_evolution(psi):
    T = kinetic_operator()
    psi_k = fft(psi)
    psi_k = np.exp(-1j * T * dt / hbar) * psi_k  # Evolve in momentum space
    psi = ifft(psi_k)  # Transform back to position space
    psi = np.exp(-1j * V * dt / hbar) * psi  # Evolve due to potential
    return psi

# Time evolution of the wavefunction
psi = psi0.copy()
wavefunctions = [psi.copy()]

for t in range(timesteps):
    psi = time_evolution(psi)
    if t % 10 == 0:
        wavefunctions.append(psi.copy())

# Create animation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, np.abs(psi)**2)
ax.set_ylim(0, 0.04)  # Adjust based on expected max probability density
ax.set_title('Wavefunction Probability Density')
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x)|²')
ax.grid()

def update(frame):
    line.set_ydata(np.abs(wavefunctions[frame])**2)  # Update the data for the line
    return line,

# Create and save the animation
ani = FuncAnimation(fig, update, frames=len(wavefunctions), blit=True, repeat=False)

# Save as GIF
# ani.save('wavefunction_animation.gif', writer=PillowWriter(fps=30))

# Optionally, save as MP4 (requires ffmpeg)
ani.save('wavefunction_animation.mp4', writer='ffmpeg', fps=30)

plt.show()