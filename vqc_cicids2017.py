"""
============================================================================
 CICIDS2017 - confronto: Random Forest vs 3 VQC vs QSVC (Quantum SVM)
----------------------------------------------------------------------------
 Modelli valutati:
   - RandomForest (baseline classica)
   - VQC con ansatz real_amplitudes    (Ry + CX reverse_linear)
   - VQC con ansatz efficient_su2      (Ry+Rz + CX reverse_linear)
   - VQC con ansatz TwoLocal custom    (Ry + CZ linear)
   - QSVC  con quantum kernel (ZZFeatureMap)

 VQC vs QSVC:
   - VQC : "rete neurale quantistica", training iterativo di parametri.
   - QSVC: SVM classica che usa un KERNEL calcolato quantisticamente
           (fidelity tra stati quantistici). Niente parametri da ottimizzare.
           Training convesso MA scala O(N^2) in memoria/tempo per la matrice
           kernel -> molto pesante su dataset grandi.

 NOTA IMPORTANTE SUL QSVC:
   Le prime versioni di qiskit-machine-learning usano internamente un
   sampler che costruisce probabilities_dict -> consumo memoria enorme con
   10 qubit e 1000+ sample (stati base = 2^10 = 1024 stringhe PER campione).
   Qui usiamo FidelityStatevectorKernel, che lavora direttamente sui
   vettori di stato senza mai costruire dizionari, ed e' molto piu' leggero.
   In piu' usiamo un SAMPLE_SIZE piccolo dedicato al QSVC, perche' la
   matrice kernel N x N rende poco pratico N grande.

 Richiede: numpy, pandas, scikit-learn, matplotlib,
           qiskit>=1.0, qiskit-machine-learning>=0.7
============================================================================
"""

import os
import glob
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

# ---- Qiskit / Qiskit Machine Learning -----------------------------------
from qiskit.circuit.library import (zz_feature_map, real_amplitudes,
                                    efficient_su2, TwoLocal)
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC

# FidelityStatevectorKernel: versione "lightweight" che NON costruisce
# probabilities_dict internamente -> niente esplosione di memoria.
# E' l'implementazione da preferire quando si lavora su simulatore
# classico con numero di qubit non banale (>= 8-10).
try:
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    _KERNEL_CLS = FidelityStatevectorKernel
    _KERNEL_NAME = "FidelityStatevectorKernel"
except ImportError:
    # Fallback per versioni piu' vecchie (rischia l'errore di memoria).
    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        _KERNEL_CLS = FidelityQuantumKernel
        _KERNEL_NAME = "FidelityQuantumKernel"
    except ImportError:
        from qiskit_machine_learning.kernels import QuantumKernel
        _KERNEL_CLS = QuantumKernel
        _KERNEL_NAME = "QuantumKernel (legacy)"


# ============================================================================
# CONFIGURAZIONE
# ============================================================================
RANDOM_SEED          = 42
DATA_PATH            = "./archive/"
SAMPLE_SIZE          = 2000          # sample per RF e VQC
QSVC_SAMPLE_SIZE     = 400            # sample SOLO per il QSVC
                                      # (kernel N x N -> scala O(N^2))
                                      # 400 -> matrice 320x320 in training
                                      #  = ~51k evaluations (ok in pochi min)
N_COMPONENTS         = 6             # qubit = componenti PCA
TRAIN_SIZE           = 0.8
BINARY_CLASSIFICATION = True
MAX_ITER_VQC         = 300
REPS_FEATURE_MAP     = 1
REPS_ANSATZ          = 3
SAVE_PLOTS           = True

# Se il QSVC continua a dare problemi su PC con poca RAM, metti a False
# per saltarlo e valutare solo RF + 3 VQC.
RUN_QSVC             = True

algorithm_globals.random_seed = RANDOM_SEED
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")


# ============================================================================
# 1. CARICAMENTO DEL DATASET
# ============================================================================
def load_cicids2017(data_path: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"Nessun file CSV trovato in '{data_path}'. "
            "Scarica il dataset da http://www.unb.ca/cic/datasets/IDS2017.html"
        )
    dfs = []
    for f in csv_files:
        print(f"    - carico {os.path.basename(f)}")
        dfs.append(pd.read_csv(f, encoding="latin1", low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


# ============================================================================
# 2. PREPROCESSING
# ============================================================================
def preprocess(df: pd.DataFrame, binary: bool = True):
    label_col = "Label"
    if label_col not in df.columns:
        raise KeyError(f"Colonna '{label_col}' non trovata.")

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"    righe rimosse per NaN/Inf: {before - len(df):,}")

    if binary:
        df = df.copy()
        df[label_col] = df[label_col].apply(
            lambda v: "BENIGN" if str(v).strip().upper() == "BENIGN" else "ATTACK"
        )

    y_raw = df[label_col].values
    X_df  = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X_df.values.astype(np.float64), y, le, X_df.columns.tolist()


# ============================================================================
# 3. SOTTOCAMPIONAMENTO BILANCIATO
# ============================================================================
def balanced_subsample(X, y, n_total, seed=123):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = n_total // len(classes)
    idx_list = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        take = min(per_class, len(idx_c))
        idx_list.append(rng.choice(idx_c, size=take, replace=False))
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ============================================================================
# 4. VALUTAZIONE UNIFICATA
# ============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test, name,
                   model_type="Classical"):
    print(f"\n    >> {name}")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - t0

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

    print(f"       train time : {train_time:8.2f} s")
    print(f"       pred  time : {pred_time:8.2f} s")
    print(f"       test  acc  : {test_acc:6.3f}")
    print(f"       F1         : {f1:6.3f}")

    return dict(
        name=name,
        type=model_type,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        train_time=float(train_time),
        pred_time=float(pred_time),
        train_acc=float(train_acc),
        test_acc=float(test_acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
    )


# ============================================================================
# 5. HELPER per il VQC
# ============================================================================
def build_vqc(feature_map, ansatz, history_list, label):
    def _cb(weights, obj_val):
        history_list.append(obj_val)
        if len(history_list) % 10 == 0:
            print(f"       [{label}] iter {len(history_list):3d} "
                  f"| obj = {obj_val:.4f}")
    return VQC(
        sampler=StatevectorSampler(),
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=MAX_ITER_VQC),
        callback=_cb,
    )


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("=" * 82)
    print(" RandomForest vs VQC (3 ansatz) vs QSVC su CICIDS2017")
    print("=" * 82)
    print(f" Config: SAMPLE_SIZE={SAMPLE_SIZE} | QSVC_SAMPLE_SIZE={QSVC_SAMPLE_SIZE}")
    print(f"         N_COMPONENTS={N_COMPONENTS} | MAX_ITER_VQC={MAX_ITER_VQC}")
    print(f"         kernel class: {_KERNEL_NAME}")

    # ---- 1. Load -----------------------------------------------------------
    print("\n[1/7] Caricamento CSV...")
    df = load_cicids2017(DATA_PATH)
    print(f"    totale flussi : {len(df):,}")

    # ---- 2. Preprocess -----------------------------------------------------
    print("\n[2/7] Preprocessing...")
    X_all, y_all, le, _ = preprocess(df, binary=BINARY_CLASSIFICATION)
    print(f"    shape dopo pulizia : {X_all.shape}")

    # ---- 3. Sottocampionamento --------------------------------------------
    print(f"\n[3/7] Sottocampionamento bilanciato a {SAMPLE_SIZE} flussi...")
    X, y = balanced_subsample(X_all, y_all, SAMPLE_SIZE, seed=RANDOM_SEED)
    counts = {cls: int((y == i).sum()) for i, cls in enumerate(le.classes_)}
    print(f"    distribuzione : {counts}")

    # ---- 4. Scaling + PCA --------------------------------------------------
    # Il fit della PCA viene fatto sul "set grande" (SAMPLE_SIZE) e poi
    # applicato sia al set per RF/VQC sia al subset piu' piccolo del QSVC.
    print(f"\n[4/7] Normalizzazione + PCA a {N_COMPONENTS} componenti...")
    scaler1 = MinMaxScaler().fit(X)
    X_s = scaler1.transform(X)
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED).fit(X_s)
    X_p = pca.transform(X_s)
    scaler2 = MinMaxScaler().fit(X_p)
    X = scaler2.transform(X_p)
    print(f"    varianza spiegata : {pca.explained_variance_ratio_.sum():.3f}")

    # ---- 5. Split train/test per RF + VQC ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"    train (RF/VQC) : {X_train.shape} | test : {X_test.shape}")

    results = []
    num_features = X_train.shape[1]

    # ---- 6a. Random Forest (baseline classica) -----------------------------
    print("\n[5/7] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                n_jobs=-1)
    results.append(evaluate_model(rf, X_train, y_train, X_test, y_test,
                                  "RandomForest", model_type="Classical"))

    feature_map = zz_feature_map(feature_dimension=num_features,
                                 reps=REPS_FEATURE_MAP)

    # ---- 6b. VQC x 3 ansatz -----------------------------------------------
    print("\n[6/7] Training 3 VQC con ansatz differenti...")

    hist_ra = []
    ansatz_ra = real_amplitudes(num_qubits=num_features, reps=REPS_ANSATZ)
    vqc_ra = build_vqc(feature_map, ansatz_ra, hist_ra, "real_amp")
    results.append(evaluate_model(vqc_ra, X_train, y_train, X_test, y_test,
                                  "VQC (real_amplitudes)",
                                  model_type="Quantum (VQC)"))

    hist_eff = []
    ansatz_eff = efficient_su2(num_qubits=num_features, reps=REPS_ANSATZ)
    vqc_eff = build_vqc(feature_map, ansatz_eff, hist_eff, "eff_su2")
    results.append(evaluate_model(vqc_eff, X_train, y_train, X_test, y_test,
                                  "VQC (efficient_su2)",
                                  model_type="Quantum (VQC)"))

    hist_tl = []
    ansatz_tl = TwoLocal(
        num_qubits=num_features,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="linear",
        reps=REPS_ANSATZ,
    )
    vqc_tl = build_vqc(feature_map, ansatz_tl, hist_tl, "two_local")
    results.append(evaluate_model(vqc_tl, X_train, y_train, X_test, y_test,
                                  "VQC (two_local Ry+CZ)",
                                  model_type="Quantum (VQC)"))

    # ---- 6c. QSVC (Quantum Support Vector Classifier) ---------------------
    if RUN_QSVC:
        print("\n[7/7] Training QSVC (Quantum Support Vector Classifier)...")
        # Costruisco un subset dedicato piu' piccolo per il QSVC, perche'
        # la matrice kernel N x N rende proibitivo usare N=1600.
        # Con N=400 -> 320 training -> kernel 320x320 = 102400 entries,
        # realisticamente qualche minuto.
        print(f"    Nota: il QSVC usa SAMPLE_SIZE ridotto = {QSVC_SAMPLE_SIZE}")
        X_q, y_q = balanced_subsample(X_all, y_all, QSVC_SAMPLE_SIZE,
                                      seed=RANDOM_SEED)
        X_q = scaler2.transform(pca.transform(scaler1.transform(X_q)))
        Xq_train, Xq_test, yq_train, yq_test = train_test_split(
            X_q, y_q, train_size=TRAIN_SIZE,
            random_state=RANDOM_SEED, stratify=y_q
        )
        print(f"    train QSVC : {Xq_train.shape} | test : {Xq_test.shape}")

        qkernel = _KERNEL_CLS(feature_map=feature_map)
        qsvc = QSVC(quantum_kernel=qkernel)

        try:
            results.append(evaluate_model(qsvc, Xq_train, yq_train,
                                          Xq_test, yq_test,
                                          "QSVC (ZZ feature_map)",
                                          model_type="Quantum (QSVC)"))
        except MemoryError as e:
            print(f"    !! QSVC fallito per memoria: {e}")
            print(f"    !! Riduci QSVC_SAMPLE_SIZE o N_COMPONENTS e riprova.")
        except Exception as e:
            print(f"    !! QSVC fallito: {type(e).__name__}: {e}")
            print(f"    !! Probabile causa: RAM insufficiente per kernel N x N")
            print(f"    !! Riduci QSVC_SAMPLE_SIZE (attuale: {QSVC_SAMPLE_SIZE})")
    else:
        print("\n[7/7] QSVC saltato (RUN_QSVC = False).")

    # ---- Tabella finale ----------------------------------------------------
    print("\n" + "=" * 88)
    print(" RISULTATI FINALI")
    print("=" * 88)
    header = (f"{'Modello':<26}{'Tipo':<17}{'N_tr':>6}"
              f"{'Pr':>7}{'Rc':>7}{'F1':>7}{'TestAcc':>9}{'Time(s)':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<26}{r['type']:<17}{r['n_train']:>6}"
              f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
              f"{r['test_acc']:>9.3f}{r['train_time']:>10.1f}")

    out_csv = "vqc_vs_rf_vs_qsvc_results.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\n    risultati esportati in: {out_csv}")

    # ---- Grafico di convergenza dei 3 VQC ---------------------------------
    if SAVE_PLOTS:
        plt.figure(figsize=(11, 5))
        if hist_ra:  plt.plot(hist_ra,  label="VQC real_amplitudes")
        if hist_eff: plt.plot(hist_eff, label="VQC efficient_su2")
        if hist_tl:  plt.plot(hist_tl,  label="VQC two_local (Ry+CZ)")
        plt.xlabel("Iterazione COBYLA")
        plt.ylabel("Objective function")
        plt.title("Convergenza dei 3 ansatz VQC su CICIDS2017")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("vqc_convergence.png", dpi=120)
        print("    grafico di convergenza: vqc_convergence.png")

    # ---- Commento conclusivo ----------------------------------------------
    print("\n" + "=" * 88)
    print(" INTERPRETAZIONE")
    print("=" * 88)
    rf_res = next((r for r in results if r["name"] == "RandomForest"), None)
    vqc_list = [r for r in results if r["type"] == "Quantum (VQC)"]
    qsvc_res = next((r for r in results if r["type"] == "Quantum (QSVC)"), None)

    if rf_res:
        print(f" Random Forest   : F1={rf_res['f1']:.3f}  "
              f"time={rf_res['train_time']:.1f}s  (N_train={rf_res['n_train']})")
    if vqc_list:
        best_vqc = max(vqc_list, key=lambda r: r["f1"])
        print(f" Miglior VQC     : {best_vqc['name']}  "
              f"F1={best_vqc['f1']:.3f}  time={best_vqc['train_time']:.1f}s")
    if qsvc_res:
        print(f" QSVC            : F1={qsvc_res['f1']:.3f}  "
              f"time={qsvc_res['train_time']:.1f}s  "
              f"(N_train={qsvc_res['n_train']} -- ridotto)")
    print("")
    print(" Ranking completo per F1:")
    for i, r in enumerate(sorted(results, key=lambda r: r["f1"],
                                 reverse=True), 1):
        print(f"   {i}. {r['name']:<26} [{r['type']:<16}] "
              f"F1={r['f1']:.3f}  time={r['train_time']:.1f}s  "
              f"(N_train={r['n_train']})")
    print("=" * 88)


if __name__ == "__main__":
    main()