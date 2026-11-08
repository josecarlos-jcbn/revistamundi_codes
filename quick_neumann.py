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
# QUICK SCHEME WITH VON NEUMANN TIME STEPPING
# =============================================================================
def solve_quick(nz):
    """Solve age equation using QUICK scheme with exact Von Neumann stability limit"""
    dz = L / nz
    z = np.linspace(0.0, L, nz + 1)

    vz = vz_dansgaard(z)
    dvz = dvz_dansgaard(z)  # B
    v_mag = np.abs(vz)  # v = -vz > 0

    # Condicao de estabilidade de Von Neumann para QUICK:
    # dt <= [v/(2*dz) - 2*B] / [B^2 - B*v/(2*dz) + 13*v^2/(8*dz^2)]
    num = (v_mag / (2.0 * dz)) - 2.0 * dvz
    den = (
        dvz**2
        - (dvz * v_mag) / (2.0 * dz)
        + (13.0 * (v_mag**2)) / (8.0 * (dz**2))
    )
    dt_local = num / den
    dt = np.min(dt_local)
    nt = int(T / dt)

    A = np.zeros((nz + 1, nt + 1))

    for n in range(nt):
        # k = 0 (base)
        Q = 1 + A[0, n] * dvz[0]
        vz_mid = vz_dansgaard(dz / 2)
        vz_0 = vz_dansgaard(0.0)
        termo_base = (-A[2, n] + 6 * A[1, n] + 3 * A[0, n]) * vz_mid - 8 * A[
            0, n
        ] * vz_0
        A[0, n + 1] = A[0, n] + Q * dt - (dt / (4 * dz)) * termo_base

        # k = 1 to nz-2 (interior)
        for k in range(1, nz - 1):
            Q = 1 + A[k, n] * dvz[k]
            vz_kph = vz_dansgaard(z[k] + dz / 2)
            vz_kmh = vz_dansgaard(z[k] - dz / 2)
            termo1 = (-A[k + 2, n] + 6 * A[k + 1, n] + 3 * A[k, n]) * vz_kph
            termo2 = (-A[k + 1, n] + 6 * A[k, n] + 3 * A[k - 1, n]) * vz_kmh
            A[k, n + 1] = A[k, n] + Q * dt - (dt / (8 * dz)) * (termo1 - termo2)

        # k = nz-1 (near surface)
        k = nz - 1
        Q = 1 + A[k, n] * dvz[k]
        vz_kph = vz_dansgaard(z[k] + dz / 2)
        vz_kmh = vz_dansgaard(z[k] - dz / 2)
        termo1 = (4 * A[k + 1, n] + 4 * A[k, n]) * vz_kph
        termo2 = (-A[k + 1, n] + 6 * A[k, n] + 3 * A[k - 1, n]) * vz_kmh
        A[k, n + 1] = A[k, n] + Q * dt - (dt / (8 * dz)) * (termo1 - termo2)

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
print("Running QUICK simulations using Von Neumann stability condition")
print("=" * 70)

for nz in resolutions:
    print(f"\n--- Resolution: nz = {nz} ---")

    z_ana = np.linspace(0.0, L, nz + 1)
    A_ana = analytical_solution_dansgaard(z_ana)

    A_quick, z_quick, dt_used = solve_quick(nz)
    error_quick = compute_relative_error(A_quick, A_ana)
    b_err_quick = basal_error(A_quick, A_ana)

    results[nz] = {
        'z': z_ana,
        'A_ana': A_ana,
        'A_quick': A_quick,
        'error_quick': error_quick,
        'basal_error_quick': b_err_quick,
        'dt': dt_used,
    }

    print(f"  Delta t (Von Neumann): {dt_used:.6e}")
    print(f"  QUICK - Basal error:   {b_err_quick * 100:.4f}%")

# =============================================================================
# PLOT 1: QUICK AGE PROFILES
# =============================================================================
colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))

fig1, ax1 = plt.subplots(figsize=(7, 8))
for i, nz in enumerate(resolutions):
    ax1.plot(
        results[nz]['A_quick'],
        results[nz]['z'],
        color=colors[i],
        linestyle='-',
        linewidth=1.5,
        label=f'QUICK (nz={nz})',
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
ax1.set_title('QUICK Method - Ice Age Profile\n(Dansgaard-Johnsen)', fontsize=14)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('QUICK_age_profile_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# PLOT 2: RELATIVE ERROR - QUICK
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(7, 8))
for i, nz in enumerate(resolutions):
    z_inner = results[nz]['z'][1:-1]
    error = results[nz]['error_quick']
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
    'QUICK Method - Relative Error Profile\n(Dansgaard-Johnsen)', fontsize=14
)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('QUICK_relative_error.png', dpi=150, bbox_inches='tight')
plt.show()
