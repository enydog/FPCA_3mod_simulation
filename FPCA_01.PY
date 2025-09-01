# ===========================
# FPCA panel académico con SEABORN (A–F)
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# --------- Estilo académico (Seaborn) ----------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --------- Reproducibilidad ----------
np.random.seed(123)

# --------- Carga de datos ----------
pd0   = pd.read_csv("GCdatcrop3.csv")
x_vec = pd.read_csv("GCdat2time.csv").iloc[:, 0].values.astype(float)

# Columnas 3:93 (0-indexed) -> 90 puntos
p = pd0.iloc[:, 3:93].copy().dropna()
assert p.shape[1] == len(x_vec), "Las columnas de p deben igualar la longitud de x_vec."

# --------- Partición train/test ----------
train_size = min(2000, len(p))
train_idx  = np.random.choice(len(p), size=train_size, replace=False)
test_idx   = np.setdiff1d(np.arange(len(p)), train_idx)
train = p.iloc[train_idx].copy()
test  = p.iloc[test_idx].copy()

# --------- Log-transform ----------
ly     = np.log(train.values)   # (n_train, T)
lytest = np.log(test.values)    # (n_test, T)
lx     = np.log(x_vec)          # grid en log-tiempo

# --------- FPCA simple ----------
class SimpleFPCA:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.mu_ = None
        self.phi_ = None
        self.lambda_ = None
        self.work_grid_ = None

    def fit(self, X, t_grid=None):
        self.work_grid_ = t_grid if t_grid is not None else np.arange(X.shape[1])
        self.mu_ = np.mean(X, axis=0)
        Xc = X - self.mu_
        self.pca.fit(Xc)
        self.phi_    = self.pca.components_.T
        self.lambda_ = self.pca.explained_variance_
        return self

    def transform(self, X):
        Xc = X - self.mu_
        return self.pca.transform(Xc)

    def predict(self, X, K=None):
        if K is None:
            K = self.n_components
        scores = self.transform(X)
        recon  = self.mu_ + scores[:, :K] @ self.phi_[:, :K].T
        return {"scores": scores, "pred_curves": recon}

# --------- Ajuste FPCA ----------
K = 3
fpca = SimpleFPCA(n_components=4).fit(ly, lx)
train_scores = fpca.transform(ly)
expl_var     = fpca.pca.explained_variance_ratio_
cum_expl     = np.cumsum(expl_var)

# --------- Reconstrucciones y residuales ----------
fit_train_log = fpca.predict(ly,     K=K)["pred_curves"]
fit_test_log  = fpca.predict(lytest, K=K)["pred_curves"]

train_ex   = np.exp(ly)
test_ex    = np.exp(lytest)
fit_train  = np.exp(fit_train_log)
fit_test   = np.exp(fit_test_log)

res_test   = 100 * (test_ex - fit_test) / fit_test
pres_test  = np.mean(res_test, axis=0)
sdres_test = np.std(res_test, axis=0)

# --------- Utilidades de formato ----------
def format_time_axis(ax):
    ax.set_xscale('log')
    ticks = np.array([5, 10, 15, 30, 60, 120, 300, 600, 1200, 3600, 7200], dtype=float)
    ticks = ticks[(ticks >= x_vec.min()) & (ticks <= x_vec.max())]
    ax.set_xticks(ticks)
    def _fmt(t):
        t = float(t)
        if t < 60:  return f"{int(t)} s"
        m = int(round(t/60))
        return f"{m} min"
    ax.set_xticklabels([_fmt(t) for t in ticks])
    ax.set_xlim(x_vec.min(), x_vec.max())

def panel_letter(ax, letter):
    ax.text(0.01, 0.98, letter, transform=ax.transAxes, ha='left', va='top',
            fontsize=13, fontweight='bold')

# --------- Figura Panel 3×2 ----------
fig, axes = plt.subplots(3, 2, figsize=(13.5, 12), constrained_layout=True)
axes = axes.ravel()

# ===== A) Curvas crudas (hasta 100) + media =====
ax = axes[0]
n_plot = min(100, train_ex.shape[0])
# DataFrame largo para seaborn con units=curve
records = []
for i in range(n_plot):
    records.extend(zip(x_vec, train_ex[i], [i]*len(x_vec)))
dfa = pd.DataFrame(records, columns=['duration', 'power', 'curve'])

sns.lineplot(data=dfa, x='duration', y='power', units='curve',
             estimator=None, linewidth=0.9, alpha=0.10, color='0.4', ax=ax)
# Media
mean_curve_ex = np.exp(fpca.mu_)
sns.lineplot(x=x_vec, y=mean_curve_ex, ax=ax, linewidth=2.0, color='black', label='Media')
format_time_axis(ax)
ax.set_ylabel('Potencia (W)')
ax.set_ylim(bottom=0)
ax.legend(frameon=False)
panel_letter(ax, 'A')
ax.set_title('Curvas individuales y función media')

# ===== B) Scree (barras) + acumulada (línea) =====
ax = axes[1]
df_scree = pd.DataFrame({
    'PC': np.arange(1, len(expl_var)+1),
    'Explained': expl_var*100,
    'Cumulative': np.cumsum(expl_var)*100
})
sns.barplot(data=df_scree, x='PC', y='Explained', ax=ax, alpha=0.8)
sns.lineplot(data=df_scree, x='PC', y='Cumulative', ax=ax, marker='o', linewidth=1.8)
ax.set_ylabel('Varianza explicada (%)')
ax.set_ylim(0, max(100, df_scree['Explained'].max()+5))
ax.set_xlabel('Componente principal')
panel_letter(ax, 'B')
ax.set_title(f'Scree. Σ₁₋₃ = {df_scree.loc[2,"Cumulative"]:.1f}%')
ax.legend(handles=[], labels=[])  # limpio leyenda redundante

# ===== C–E) Abanicos ±2·SD para FPC1–FPC3 =====
for j in range(3):
    ax = axes[2+j]
    mean_log = fpca.mu_
    phi_j    = fpca.phi_[:, j]
    sd_j     = train_scores[:, j].std()
    n_vals   = np.linspace(-2.0*sd_j, 2.0*sd_j, 21)

    # DataFrame largo para seaborn
    rec = []
    for n in n_vals:
        curve = np.exp(mean_log + phi_j*n)
        rec.extend(zip(x_vec, curve, [n]*len(x_vec)))
    dfj = pd.DataFrame(rec, columns=['duration', 'power', 'score'])

    # Paleta homogénea (no legend para estética limpia)
    pal = sns.color_palette("Greys", n_colors=len(n_vals))
    # Mapeo discreto de score a color (ordenado)
    dfj['_rank'] = pd.cut(dfj['score'], bins=len(n_vals), labels=False, include_lowest=True)

    # Media
    sns.lineplot(x=x_vec, y=np.exp(mean_log), ax=ax, linewidth=2.0, color='black')

    # Abanico
    # Usamos hue por _rank para gradiente; sin leyenda
    sns.lineplot(data=dfj, x='duration', y='power', hue='_rank', palette=pal,
                 legend=False, estimator=None, units=None, linewidth=0.7, alpha=0.6, ax=ax)

    format_time_axis(ax)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Potencia (W)')
    panel_letter(ax, chr(ord('C')+j))
    ax.set_title(f'FPC{j+1}: variación ±2·SD (score)')

# ===== F) Residual (%) en test con banda ±2·SD =====
ax = axes[5]
df_res = pd.DataFrame({'duration': x_vec,
                       'mean_res': pres_test,
                       'lo': pres_test - 2*sdres_test,
                       'hi': pres_test + 2*sdres_test})
sns.lineplot(data=df_res, x='duration', y='mean_res', ax=ax, color='black', marker='o', markersize=3, linewidth=0.9)
ax.fill_between(df_res['duration'], df_res['lo'], df_res['hi'], alpha=0.25, color='0.6', linewidth=0)
format_time_axis(ax)
ax.axhline(0, linestyle='--', color='black', linewidth=0.8, alpha=0.6)
ax.set_ylabel('Residual (%)')
ax.set_xlabel('Duración')
ax.set_ylim(-30, 30)
panel_letter(ax, 'F')
ax.set_title('Residual (%) por duración (test)  ±2·SD')

# --------- Toques finales ----------
fig.suptitle('FPCA de curvas MMP – Panel académico (Seaborn)', y=1.02,
             fontsize=15, fontweight='bold')
sns.despine(fig)
plt.savefig("fpca_panel_seaborn.png", dpi=300, bbox_inches="tight")
plt.show()

# --------- Métricas clave ----------
print("\n== Resumen FPCA ==")
print(f"Curvas (train): {ly.shape[0]}  |  Puntos de tiempo: {ly.shape[1]}")
print(f"Curvas (test):  {lytest.shape[0]}")
print("Varianza explicada por PC (1–4):", np.round(expl_var[:4]*100, 2), "%")
print(f"Suma PC1–PC3: {cum_expl[2]*100:.2f}%")
print(f"Residual test: media(abs)={np.mean(np.abs(pres_test)):.2f}%  "
      f"SD_promedio={np.mean(sdres_test):.2f}%")
print("Figura guardada como: fpca_panel_seaborn.png")
