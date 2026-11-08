# =============================================================================
# COMPARISON: EDC vs NUMERICAL METHODS
# PROFILES: DANSGAARD-JOHNSEN, LLIBOUTRY, AND HYBRID (DJ + LL)
# VON NEUMANN ANALYSIS + 3 METHODS
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAÇÃO DE FONTE
# =============================================================================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"], 
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "mathtext.fontset": "cm"
})

# Cria a pasta unificada para salvar as figuras
os.makedirs('figuras_tres_perfis', exist_ok=True)

# =============================================================================
# 1. PARAMETERS
# =============================================================================
L = 1.0
T = 1000.0
nz = 100
dz = L / nz

# =============================================================================
# 2. IMPORT EDC DATA
# =============================================================================
arquivo = "EDC_AICC2012_chron.tab"

skip = [i for i, line in enumerate(open(arquivo, encoding='latin-1'))
        if "*/" in line][0] + 1

dados_edc = pd.read_csv(arquivo, sep='\t', skiprows=skip)

H_real = dados_edc['Depth ice/snow [m]'].max()
a_real = dados_edc['Acc rate ice per year [m/a]'].mean()
tau = H_real / a_real

# Exact adimensionalization from reference
z_edc = 1 - (dados_edc['Depth ice/snow [m]'] / H_real)
age_edc_adim = (dados_edc['Age [ka BP]'] * 1000) / tau

# =============================================================================
# 3. VELOCITY PROFILES DEFINITIONS
# =============================================================================

# --- Profile 1: Dansgaard-Johnsen (DJ) ---
Z_AST_DJ, VZB_DJ = 0.25, -2.5e-3
c1_DJ = (2*(1 + VZB_DJ)) / (2 - Z_AST_DJ)
c2_DJ = (Z_AST_DJ + 2*VZB_DJ) / (2 - Z_AST_DJ)
c3_DJ = (1 + VZB_DJ) / (Z_AST_DJ*(2 - Z_AST_DJ))
c4_DJ = -VZB_DJ

def vz_DJ(z_val):
    z_val = np.asarray(z_val, dtype=float)
    return np.where(z_val >= Z_AST_DJ, -c1_DJ*z_val + c2_DJ, -c3_DJ*z_val**2 - c4_DJ)

def dvz_DJ(z_val):
    z_val = np.asarray(z_val, dtype=float)
    return np.where(z_val >= Z_AST_DJ, -c1_DJ*np.ones_like(z_val), -2*c3_DJ*z_val)

# --- Profile 2: Lliboutry (LL, p = 3.6) ---
P_LL = 3.6

def vz_LL(z_val):
    z_val = np.asarray(z_val, dtype=float)
    term1 = (P_LL + 2) / (P_LL + 1) * (1 - z_val)
    term2 = (1 / (P_LL + 1)) * (1 - z_val)**(P_LL + 2)
    return -1.0 * (1.0 - term1 + term2)

def dvz_LL(z_val):
    z_val = np.asarray(z_val, dtype=float)
    constante = (P_LL + 2) / (P_LL + 1)
    return -constante * (1 - (1 - z_val)**(P_LL + 1))

# --- Profile 3: Hybrid (DJ + LL) ---
Z_STAR_HIB = 0.2
_offset_LL = (float(vz_DJ(np.array([Z_STAR_HIB]))[0]) 
              - float(vz_LL(np.array([Z_STAR_HIB]))[0]))

def vz_DJLL(z_val):
    z_val = np.asarray(z_val, dtype=float)
    return np.where(z_val >= Z_STAR_HIB,
                    vz_DJ(z_val),
                    vz_LL(z_val) + _offset_LL)

def dvz_DJLL(z_val):
    z_val = np.asarray(z_val, dtype=float)
    return np.where(z_val >= Z_STAR_HIB,
                    dvz_DJ(z_val),
                    dvz_LL(z_val))

# =============================================================================
# 4. NUMERICAL METHODS
# =============================================================================
def passo_UP1(A, dt, dvz_grid, vz_nodes, vz_half, vz_0):
    Q = 1.0 + A * dvz_grid
    Anew = np.empty_like(A)
    Anew[0] = A[0] + Q[0]*dt - (2.0*dt/dz)*(A[1]*vz_half[0] - A[0]*vz_nodes[0])
    Anew[1:nz] = (
        A[1:nz] + Q[1:nz]*dt
        - (dt/dz)*(A[2:nz+1]*vz_half[1:nz] - A[1:nz]*vz_half[0:nz-1])
    )
    Anew[nz] = 0.0
    return Anew

def passo_UP2(A, dt, dvz_grid, vz_nodes, vz_half, vz_0):
    Q = 1.0 + A * dvz_grid
    Anew = np.empty_like(A)
    Anew[0] = A[0] + Q[0]*dt - (dt/dz)*(
        (-A[2] + 3.0*A[1])*vz_half[0] - 2.0*A[0]*vz_nodes[0]
    )
    k = np.arange(1, nz - 1)
    termo = (
        (-A[k+2] + 3.0*A[k+1])*vz_half[k]
        - (-A[k+1] + 3.0*A[k])*vz_half[k-1]
    )
    Anew[k] = A[k] + Q[k]*dt - (dt/(2.0*dz))*termo
    k = nz - 1
    termo_s = (
        (A[nz] + A[k])*vz_half[k]
        - (-A[nz] + 3.0*A[k])*vz_half[k-1]
    )
    Anew[k] = A[k] + Q[k]*dt - (dt/(2.0*dz))*termo_s
    Anew[nz] = 0.0
    return Anew

def passo_QUICK(A, dt, dvz_grid, vz_nodes, vz_half, vz_0):
    Q = 1.0 + A * dvz_grid
    Anew = np.empty_like(A)
    termo_b = (
        (-A[2] + 6.0*A[1] + 3.0*A[0])*vz_half[0]
        - 8.0*A[0]*vz_nodes[0]
    )
    Anew[0] = A[0] + Q[0]*dt - (dt/(4.0*dz))*termo_b
    k = np.arange(1, nz - 1)
    termo1 = (-A[k+2] + 6.0*A[k+1] + 3.0*A[k])*vz_half[k]
    termo2 = (-A[k+1] + 6.0*A[k] + 3.0*A[k-1])*vz_half[k-1]
    Anew[k] = A[k] + Q[k]*dt - (dt/(8.0*dz))*(termo1 - termo2)
    k = nz - 1
    termo1 = (4.0*A[k+1] + 4.0*A[k])*vz_half[k]
    termo2 = (-A[k+1] + 6.0*A[k] + 3.0*A[k-1])*vz_half[k-1]
    Anew[k] = A[k] + Q[k]*dt - (dt/(8.0*dz))*(termo1 - termo2)
    Anew[nz] = 0.0
    return Anew

PASSOS = {"UP1": passo_UP1, "UP2": passo_UP2, "QUICK": passo_QUICK}

# =============================================================================
# 5. SOLVER (with von Neumann)
# =============================================================================
def resolver(metodo, vz_func, dvz_func, fallback_up1=False):
    z = np.linspace(0.0, L, nz + 1)
    dvz_grid = dvz_func(z)
    vz_nodes = vz_func(z)
    z_half   = z[:-1] + dz/2
    vz_half  = vz_func(z_half)
    vz_0     = float(vz_nodes[0])

    v_mag = np.abs(vz_nodes)

    # --- Von Neumann Analysis ---
    if metodo == "UP1":
        dt_local = -2.0 / (dvz_grid + (2.0 * vz_nodes) / dz)
    elif metodo == "UP2":
        dt_local = 2.0 / ((4.0 * v_mag / dz) - dvz_grid)
    elif metodo == "QUICK":
        num_q = (v_mag / (2.0 * dz)) - 2.0 * dvz_grid
        den_q = (dvz_grid**2 - (dvz_grid * v_mag) / (2.0 * dz) + (13.0 * v_mag**2) / (8.0 * dz**2))
        dt_local = num_q / den_q
    else:
        raise ValueError(metodo)

    # Fallback for UP1
    if metodo == "UP1" and fallback_up1:
        dt = 0.5 * dz
        print(f'  -> UP1: Using fixed dt (0.5*dz) = {dt:.4e}')
    else:
        dt = np.min(dt_local)
        print(f'  -> {metodo}: Using Von Neumann dt = {dt:.4e}')

    nt = int(T / dt)

    passo = PASSOS[metodo]
    A = np.zeros(nz + 1)
    for _ in range(nt):
        A = passo(A, dt, dvz_grid, vz_nodes, vz_half, vz_0)

    return A, z

# =============================================================================
# 6. SIMULATIONS
# =============================================================================
METODOS = ["UP1", "UP2", "QUICK"]
PERFIS = {
    "DJ":   {"vz": vz_DJ,   "dvz": dvz_DJ,   "fallback": False, "label": "Dansgaard-Johnsen"},
    "LL":   {"vz": vz_LL,   "dvz": dvz_LL,   "fallback": True,  "label": "Lliboutry (p = 3.6)"},
    "DJLL": {"vz": vz_DJLL, "dvz": dvz_DJLL, "fallback": True,  "label": "Hybrid Dansgaard-Johnsen + Lliboutry"}
}

resultados = {}

for nome, info in PERFIS.items():
    resultados[nome] = {}
    for metodo in METODOS:
        print(f'\nRunning {nome} - {metodo}...')
        A, z = resolver(metodo, info["vz"], info["dvz"], fallback_up1=info["fallback"])
        resultados[nome][metodo] = A

# =============================================================================
# 7. EDC INTERPOLATION AND RELATIVE ERROR
# =============================================================================
A_edc_interp = np.interp(z, z_edc[::-1], age_edc_adim[::-1])

def compute_relative_error(A_num):
    return np.abs((A_num[1:-1] - A_edc_interp[1:-1]) / A_edc_interp[1:-1])

# =============================================================================
# 8. INDIVIDUAL PLOTS
# =============================================================================
cores = {"UP1": 'blue', "UP2": 'green', "QUICK": 'red'}
estilos = {"UP1": '--', "UP2": '-.', "QUICK": '-'}

for nome, info in PERFIS.items():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(age_edc_adim, z_edc, 'ko', markersize=3, alpha=0.5, label='EDC')
    
    for metodo in METODOS:
        ax.plot(resultados[nome][metodo], z, color=cores[metodo], linestyle=estilos[metodo],
                 linewidth=2, label=metodo)
    
    ax.set_xscale('log')
    ax.set_xlabel('Adimensional Age ($A / \\tau$)', fontsize=16)
    ax.set_ylabel('Normalized Height z', fontsize=16)
    ax.set_title(f'{info["label"]} Profile', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=14)
    ax.set_ylim(0, 1.01)
    ax.set_xlim(1e-3, 100)
    plt.tight_layout()
    plt.savefig(f'figuras_tres_perfis/01_age_{nome}.png', dpi=300, bbox_inches='tight')
    print(f"  [saved] figuras_tres_perfis/01_age_{nome}.png")
    plt.show()

# =============================================================================
# 9. COMBINED PLOT
# =============================================================================
fig_comb, (ax_vel, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8))

ax_vel.plot(vz_DJ(z), z, lw=2.5, color='#1f77b4', linestyle='--', 
            marker='s', markevery=5, markersize=5, label=PERFIS["DJ"]["label"])
ax_vel.plot(vz_LL(z), z, lw=2.5, color='#ff7f0e', linestyle='-.', 
            marker='^', markevery=5, markersize=5, label=PERFIS["LL"]["label"])
ax_vel.plot(vz_DJLL(z), z, lw=2.5, color='#2ca02c', linestyle=':', 
            marker='o', markevery=5, markersize=5, label=PERFIS["DJLL"]["label"])
ax_vel.axhline(y=Z_STAR_HIB, color='red', linestyle=':', linewidth=1.5, label=r'$z^* = 0.2$')

ax_vel.set_xlabel(r'Vertical Velocity v_z', fontsize=16)
ax_vel.set_ylabel('Normalized Height z', fontsize=16)
ax_vel.set_title('Velocity Profiles', fontsize=18)
ax_vel.tick_params(axis='both', labelsize=14)
ax_vel.grid(True, alpha=0.4)
ax_vel.legend(loc='best', fontsize=14)
ax_vel.set_ylim(0, 1.01)

v_DJ_plot   = vz_DJ(z)
v_LL_plot   = vz_LL(z)
v_DJLL_plot = vz_DJLL(z)
mascara = z <= 0.3

ax_zoom.semilogx(np.abs(v_DJ_plot[mascara]), z[mascara], label='DJ', lw=3, color='#1f77b4', linestyle='--')
ax_zoom.semilogx(np.abs(v_LL_plot[mascara]), z[mascara], label='LL', lw=2, color='#ff7f0e', linestyle='-.')
ax_zoom.semilogx(np.abs(v_DJLL_plot[mascara]), z[mascara], label='Hybrid', lw=2, color='#2ca02c', linestyle=':')
ax_zoom.axhline(y=Z_STAR_HIB, color='red', linestyle='--', alpha=0.5, label=r'$z^* = 0.2$')

ax_zoom.set_xlabel(r'Absolute Velocity |v_z|', fontsize=16)
ax_zoom.set_ylabel('Normalized Height z', fontsize=16)
ax_zoom.set_title('Basal Region', fontsize=18)
ax_zoom.tick_params(axis='both', labelsize=14)
ax_zoom.grid(True, which='both', alpha=0.3)
ax_zoom.legend(loc='best', fontsize=14)
ax_zoom.set_ylim(0, 0.3)

plt.tight_layout()
plt.savefig('figuras_tres_perfis/02_velocity_and_zoom.png', dpi=300, bbox_inches='tight')
print("  [saved] figuras_tres_perfis/02_velocity_and_zoom.png")
plt.show()

# =============================================================================
# 10. COMBINED PLOT: RELATIVE ERROR
# =============================================================================
fig_err, ax_err = plt.subplots(figsize=(10, 10))
z_inner = z[1:-1]

estilos_perfil = {
    "DJ":   '-',
    "LL":   '--',
    "DJLL": ':'
}
alpha_perfil = {
    "DJ":   0.5,
    "LL":   0.7,
    "DJLL": 0.9
}

for nome, info in PERFIS.items():
    for metodo in METODOS:
        erro = compute_relative_error(resultados[nome][metodo])
        ax_err.semilogy(erro, z_inner, 
                        color=cores[metodo], 
                        linestyle=estilos_perfil[nome],
                        alpha=alpha_perfil[nome],
                        linewidth=2, 
                        label=f'{info["label"]} - {metodo}')

ax_err.set_xlabel('Relative Error $|A_{num} - A_{EDC}| / A_{EDC}$', fontsize=16)
ax_err.set_ylabel('Normalized Height z', fontsize=16)
ax_err.set_title('Relative Error vs EDC', fontsize=18)
ax_err.tick_params(axis='both', labelsize=14)
ax_err.grid(True, which='both', alpha=0.3)
ax_err.set_ylim(0, 1.01)
ax_err.set_xlim(1e-5, 10)

ax_err.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig('figuras_tres_perfis/03_relative_error_all.png', dpi=300, bbox_inches='tight')
print("  [saved] figuras_tres_perfis/03_relative_error_all.png")
plt.show()
plt.savefig('figuras_tres_perfis/04_velocity_difference.png', dpi=300, bbox_inches='tight')
print("  [saved] figuras_tres_perfis/04_velocity_difference.png")
plt.show()

print("\nAll figures saved successfully in 'figuras_tres_perfis/'.")
