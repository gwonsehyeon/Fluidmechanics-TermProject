import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import iv


st.set_page_config(page_title="PR Dispensing Jet Simulator", layout="wide")

st.title("💧 PR Dispensing Jet: Rayleigh-Plateau Instability Simulator")
st.markdown("Chemical Engineering Fluid Mechanics Term Project - School of Chemical Engineering, SKKU")


st.sidebar.header("⚙️ Process & Material Parameters")
r0_um = st.sidebar.slider("Nozzle Radius r₀ (μm)", 10.0, 500.0, 100.0)
r0 = r0_um * 1e-6
U = st.sidebar.slider("Ejection Velocity U (m/s)", 0.5, 10.0, 2.0)
rho = st.sidebar.slider("Fluid Density ρ (kg/m³)", 800.0, 1200.0, 1000.0)

mu = st.sidebar.slider("Dynamic Viscosity μ (Pa·s)", 0.0, 100.0, 0.1, format="%.2f") 
gamma = st.sidebar.slider("Surface Tension γ (N/m)", 0.01, 0.07, 0.03)
L_cm = st.sidebar.slider("Substrate Distance L (cm)", 1.0, 20.0, 5.0)
L = L_cm * 1e-2

eps0 = 0.01 * r0 


x = np.linspace(0.01, 1.0, 500) 


omega_0_sq = (gamma / (rho * r0**3)) * (x * iv(1, x) / iv(0, x)) * (1 - x**2)
omega_0_sq = np.maximum(omega_0_sq, 0)


viscous_damping = (3 * mu * x**2) / (rho * r0**2)


omega = 0.5 * (-viscous_damping + np.sqrt(viscous_damping**2 + 4 * omega_0_sq))


max_idx = np.argmax(omega)
x_max = x[max_idx]
omega_max = omega[max_idx]
lambda_max = 2 * np.pi * r0 / x_max
tb = (1 / omega_max) * np.log(r0 / eps0)
zb = U * tb


col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Core Interactive View: Jet Profile")
    z = np.linspace(0, L, 1000)
    k_max = x_max / r0
    

    envelope = eps0 * np.exp(omega_max * (z / U))
    r_profile = r0 - envelope
    broken_idx = np.where(r_profile <= 0)[0]
    
    fig_jet = go.Figure()
    
    if len(broken_idx) > 0: 
        break_z = z[broken_idx[0]]
        z_intact = z[:broken_idx[0]]
        r_intact = r0 - eps0 * np.exp(omega_max * (z_intact / U)) * np.cos(k_max * z_intact)
        

        fig_jet.add_trace(go.Scatter(x=z_intact*100, y=r_intact*1e6, fill='tozeroy', mode='lines', line_color='blue', name='Intact PR Jet'))
        fig_jet.add_trace(go.Scatter(x=z_intact*100, y=-r_intact*1e6, fill='tozeroy', mode='lines', line_color='blue', showlegend=False))
        

        z_drops = z[broken_idx[0]::50] 
        fig_jet.add_trace(go.Scatter(x=z_drops*100, y=np.zeros_like(z_drops), mode='markers', marker=dict(size=12, color='blue'), name='Droplets'))
        
        st.error(f"⚠️ Breakup occurs at {break_z*100:.2f} cm! The jet will NOT reach the substrate intact.")
    else: 
        r_intact = r0 - eps0 * np.exp(omega_max * (z / U)) * np.cos(k_max * z)
        fig_jet.add_trace(go.Scatter(x=z*100, y=r_intact*1e6, fill='tozeroy', mode='lines', line_color='teal', name='Intact PR Jet'))
        fig_jet.add_trace(go.Scatter(x=z*100, y=-r_intact*1e6, fill='tozeroy', mode='lines', line_color='teal', showlegend=False))
        st.success(f"✅ The jet safely reaches the substrate at {L_cm} cm without breaking up.")


    fig_jet.add_vline(x=L_cm, line_dash="dash", line_color="red", annotation_text="Substrate")
    fig_jet.update_layout(xaxis_title="Distance from Nozzle z (cm)", yaxis_title="Radius r (μm)", height=400)
    st.plotly_chart(fig_jet, width="stretch")

with col2:
    st.subheader("2. Validation View: Dispersion Relation")
    fig_val = go.Figure()

    fig_val.add_trace(go.Scatter(x=x, y=omega, mode='lines', line=dict(color='purple', width=3), name='Growth Rate ω(k)'))
    fig_val.add_trace(go.Scatter(x=[x_max], y=[omega_max], mode='markers', marker=dict(size=12, color='red'), name=f'Max: x={x_max:.3f}'))
    

    fig_val.update_layout(xaxis_title="Dimensionless Wavenumber (k·r₀)", yaxis_title="Growth Rate ω(k) (1/s)", height=400)
    st.plotly_chart(fig_val, width="stretch")


st.markdown("---")
st.subheader("📊 Key Process Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Max Growth Rate (ω_max)", f"{omega_max:.1f} s⁻¹")
m2.metric("Most Unstable Wavelength (λ_max)", f"{lambda_max*1e6:.1f} μm")
m3.metric("Breakup Time (t_b)", f"{tb*1000:.2f} ms")
m4.metric("Predicted Breakup Distance (z_b)", f"{zb*100:.2f} cm")
