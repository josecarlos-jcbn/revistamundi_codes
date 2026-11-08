import matplotlib.pyplot as plt
import numpy as np
from numpy import arctan, log, sqrt

# =============================================================================
# PHYSICAL PARAMETERS AND DISCRETIZATION
# =============================================================================
L = 1.0  # dimensionless height
T = 1000.0  # dimensionless time for steady state

# Dansgaard-Johnsen profile parameters
z_star = 0.25
vz_b = -2.5e-3  # basal velocity
vz_1 = -1.0  # surface velocity

# Profile constants
c1 = (2 * (1 + vz_b)) / (2 - z_star)
c2 = (z_star + 2 * vz_b) / (2 - z_star)
c3 = (1 + vz_b) / (z_star * (2 - z_star))
c4 = -vz_b


# =============================================================================
# DANSGAARD-JOHNSEN PROFILE FUNCTIONS
# =============================================================================
def vz_dansgaard(z_val):
    """Vertical velocity - Dansgaard-Johnsen profile"""
    if np.isscalar(z_val):
        if z_val >= z_star:
            return -c1 * z_val + c2
        else:
            return -c3 * (z_val**2) - c4
    else:
        vals = np.zeros_like(z_val)
        for i, x in enumerate(z_val):
            if x >= z_star:
                vals[i] = -c1 * x + c2
            else:
                vals[i] = -c3 * (x**2) - c4
        return vals


def dvz_dansgaard(z_val):
    """Derivative of vertical velocity"""
    if np.isscalar(z_val):
        if z_val >= z_star:
            return -c1
        else:
            return -2 * c3 * z_val
    else:
        vals = np.zeros_like(z_val)
        for i, x in enumerate(z_val):
            if x >= z_star:
                vals[i] = -c1
            else:
                vals[i] = -2 * c3 * x
        return vals


# =============================================================================
# ANALYTICAL SOLUTION
# =============================================================================
def analytical_solution_dansgaard(z):
    """Analytical solution of the age equation for Dansgaard-Johnsen profile"""
    A_ana = np.zeros_like(z)
    for i, x in enumerate(z):
        if x > z_star:
            A_ana[i] = (1 / c1) * log(1 / (c1 * x - c2))
        else:
            A_ana[i] = 1 / (sqrt(c3 * c4)) * (
                arctan(sqrt(c3 / c4) * z_star) - arctan(sqrt(c3 / c4) * x)
            ) + (1 / c1) * log(1 / (c1 * z_star - c2))
    return A_ana


# =============================================================================
# UP2 SCHEME WITH VON NEUMANN TIME STEPPING
# =============================================================================
def solve_up2(nz):
    """Solve age equation using UP2 scheme with exact Von Neumann stability limit"""
    dz = L / nz
    z = np.linspace(0.0, L, nz + 1)

    vz = vz_dansgaard(z)
    dvz = dvz_dansgaard(z)  # B
    v_mag = np.abs(vz)  # v = -vz > 0

    # Condicao de estabilidade de Von Neumann para UP2:
    # dt <= 2 / (4*v/dz - B)
    dt_local = 2.0 / ((4.0 * v_mag / dz) - dvz)
    dt = np.min(dt_local)
    nt = int(T / dt)

    A = np.zeros((nz + 1, nt + 1))

    for n in range(nt):
        # k = 0 (base)
        Q = 1 + A[0, n] * dvz[0]
        vz_mid = vz_dansgaard(dz / 2)
        vz_0 = vz_dansgaard(0.0)
        A[0, n + 1] = A[0, n] + Q * dt - (dt / dz) * (
            (-A[2, n] + 3 * A[1, n]) * vz_mid - 2 * A[0, n] * vz_0
        )

        # k = 1 to nz-2 (interior)
        for k in range(1, nz - 1):
            Q = 1 + A[k, n] * dvz[k]
            vz_kph = vz_dansgaard(z[k] + dz / 2)
            vz_kmh = vz_dansgaard(z[k] - dz / 2)
            A[k, n + 1] = A[k, n] + Q * dt - (dt / (2 * dz)) * (
                (-A[k + 2, n] + 3 * A[k + 1, n]) * vz_kph
                - (-A[k + 1, n] + 3 * A[k, n]) * vz_kmh
            )

        # k = nz-1 (near surface)
        k = nz - 1
        Q = 1 + A[k, n] * dvz[k]
        vz_kph = vz_dansgaard(z[k] + dz / 2)
        vz_km3h = vz_dansgaard(z[k] - dz / 2)
        A[k, n + 1] = A[k, n] + Q * dt - (dt / (2 * dz)) * (
            (A[nz, n] + A[k, n]) * vz_kph
            - (-A[nz, n] + 3 * A[k, n]) * vz_km3h
        )

        # k = nz (surface)
        A[nz, n + 1] = 0.0

    return A[:, -1], z, dt


# =============================================================================
# ERROR CALCULATION FUNCTIONS
# =============================================================================
def compute_relative_error(A_num, A_ana):
    """Compute relative error (excluding surface and base)"""
    return np.abs((A_num[1:-1] - A_ana[1:-1]) / A_ana[1:-1])


def basal_error(A_num, A_ana):
    """Compute relative error at the base"""
    return (A_num[0] - A_ana[0]) / A_ana[0]


# =============================================================================
# EXECUTION FOR DIFFERENT RESOLUTIONS
# =============================================================================
resolutions = [20, 40, 60, 80, 100]
results = {}

print("=" * 70)
print("Running UP2 simulations using Von Neumann stability condition")
print("=" * 70)

for nz in resolutions:
    print(f"\n--- Resolution: nz = {nz} ---")

    z_ana = np.linspace(0.0, L, nz + 1)
    A_ana = analytical_solution_dansgaard(z_ana)

    A_up2, z_up2, dt_used = solve_up2(nz)
    error_up2 = compute_relative_error(A_up2, A_ana)
    b_err_up2 = basal_error(A_up2, A_ana)

    results[nz] = {
        'z': z_ana,
        'A_ana': A_ana,
        'A_up2': A_up2,
        'error_up2': error_up2,
        'basal_error_up2': b_err_up2,
        'dt': dt_used,
    }

    print(f"  Delta t (Von Neumann): {dt_used:.6e}")
    print(f"  UP2 - Basal error:     {b_err_up2 * 100:.4f}%")

# =============================================================================
# PLOT 1: UP2 AGE PROFILES
# =============================================================================
colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))

fig1, ax1 = plt.subplots(figsize=(7, 8))
for i, nz in enumerate(resolutions):
    ax1.plot(
        results[nz]['A_up2'],
        results[nz]['z'],
        color=colors[i],
        linestyle='-',
        linewidth=1.5,
        label=f'UP2 (nz={nz})',
    )
ax1.plot(
    results[resolutions[0]]['A_ana'],
    results[resolutions[0]]['z'],
    'k--',
    linewidth=2.5,
    label='Analytical Solution',
)
ax1.set_xlabel('Ice Age A(z)', fontsize=12)
ax1.set_ylabel('Normalized Height z', fontsize=12)
ax1.set_title('UP2 Method - Ice Age Profile\n(Dansgaard-Johnsen)', fontsize=14)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('UP2_age_profile_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# PLOT 2: RELATIVE ERROR - UP2
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(7, 8))
for i, nz in enumerate(resolutions):
    z_inner = results[nz]['z'][1:-1]
    error = results[nz]['error_up2']
    ax2.semilogy(
        error,
        z_inner,
        color=colors[i],
        linestyle='-',
        linewidth=1.5,
        label=f'nz={nz}',
    )
ax2.set_xlabel('Relative Error |r(z)|', fontsize=12)
ax2.set_ylabel('Normalized Height z', fontsize=12)
ax2.set_title(
    'UP2 Method - Relative Error Profile\n(Dansgaard-Johnsen)', fontsize=14
)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('UP2_relative_error.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# TABLE OF BASAL ERRORS
# =============================================================================
print("\n" + "=" * 70)
print("TABLE OF BASAL ERRORS & TIME STEPS (UP2)")
print("=" * 70)
print(f"{'nz':<6} {'Delta t':<15} {'Basal Error (%)':<20}")
print("-" * 45)

for nz in resolutions:
    up2_err = results[nz]['basal_error_up2'] * 100
    dt_val = results[nz]['dt']
    print(f"{nz:<6} {dt_val:<15.6e} {up2_err:<20.6f}")

print("\n" + "=" * 70)
print("Simulations completed!")
print("=" * 70)