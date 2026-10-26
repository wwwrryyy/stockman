# Product Requirements Document (PRD): Multi-Modal Quantitative Alpha & Risk Engine

## Document Metadata
| Field | Value |
| :--- | :--- |
| **Project Code** | QARE-MM |
| **Document Version** | 1.3 (Residualized, Sample-Efficient, PIT-Strict, Gate-Enforced) |
| **System Classification** | Cross-Sectional & Multi-Modal Statistical Arbitrage Swing-Trading System |
| **Target Hardware** | Apple Silicon M5 Air (24 GB Unified Memory, Fanless Architecture) |
| **Lookback Horizon** | Rolling 126 Trading Days ($T = 126$, ~6 Calendar Months) |
| **Prediction Horizon** | 5 to 20 Trading Days (Residualized Triple-Barrier with Calendar-Date Purging) |
| **Asset Universe** | 250 Liquid US Mid/Large-Cap Equities (S&P 500 Rank 100–500 + S&P 400 MidCap, $> \$25\text{M}$ ADV) |
| **Frictional Hurdle** | 15 bps (0.15%) round-trip penalty per trade (slippage, spread, and brokerage) |

---

## 1. Executive Summary & System Philosophy

QARE-MM v1.3 is an institutional-grade cross-sectional swing-trading and portfolio risk-management engine. Designed specifically to avoid the sample starvation, retail indicator decay, and data-scraping traps of academic multi-modal models, QARE-MM targets **market- and sector-residualized idiosyncratic alpha** across 250 liquid mid-to-large-cap US equities.

The system synthesizes four high-signal, strictly Point-in-Time (PIT) information channels over a rolling 126-trading-day window:
1. **Cross-Sectional Price, Flow & Intraday Microstructure** (Overnight/Intraday split, Beta-Residualized Momentum, Amihud Illiquidity, and Intraday VWAP/Volume skew)
2. **SEC EDGAR Insider & Filing Catalysts** (Form 4 Open-Market Insider Cluster scoring, `difflib` 10-Q/10-K Item 1A/MD&A redlines scored via local Llama-3.1-8B, and Post-Earnings Announcement Drift)
3. **Rolling Statistical Lead-Lag & GICS Peer Graphs** (Point-in-Time PyG Graph Attention Network with SEC XBRL `filed`-date fundamentals)
4. **Macro Volatility Regime Filtering & Smart-Money Divergence** (Forward-Filtered 3-State Gaussian HMM + Signed Volume-Sentiment Divergence)

The architecture treats **Sample-Weighted Gradient Boosted Decision Trees (LightGBM/CatBoost)** and a **Compact Temporal Neural Network (`hidden_size=32`)** as co-equal model candidates. Whichever architecture wins the out-of-sample walk-forward ablation gates emits **raw uncalibrated logits** into an out-of-sample **Temperature Scaler** and a **Multi-Outcome Fractional Kelly Criterion** risk engine.

### Core Engineering Principles
* **Idiosyncratic Target Isolation:** Never predict raw stock returns (which are ~65% macro and sector beta). Label and predict returns stripped of rolling SPY and GICS Sector ETF beta.
* **Simplicity & Falsifiability:** Every feature group and architectural branch must survive ablation against a simple tabular baseline net of 15 bps friction.
* **Strict Point-in-Time (PIT) Integrity:** Events, SEC filings (Form 4, 10-Q, XBRL fundamentals), and news released after 16:00 EST on day $t$ are strictly stamped as day $t+1$. Restated back-dated fundamentals and static look-ahead graphs are prohibited.
* **Apple Silicon Ergonomics:** Right-sized latent dimensions (`32` channels) and offline Parquet caching allow full walk-forward sweeps within the 24 GB Unified Memory and thermal envelope of the fanless M5 MacBook Air.

---

## 2. Phase Gating & Stop Rules

Execution follows a sequential 5-gate pipeline (Phases 0–4). If any phase fails its gate, work stops to remediate or prune complexity before proceeding.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE GATING PIPELINE (v1.3)                    │
└────────────────────────────────────────────────────────────────────────┘
  Phase 0: Lock Spec v1.3  ──► [GATE 0: All Tensors, PIT Rules & Universe Locked]
            │
            ▼
  Phase 1: Residualized    ──► [GATE 1: Passes Shuffled/Lagged Sanity Checks;
           Backbone                      Beats Baselines Net of 15 bps OOS]
            │
            ▼
  Phase 2: Regime & Peer   ──► [GATE 2: Ablation Shows Filtered HMM & Rolling
           Graph                         Peer GAT Improve Sharpe / Drawdown]
            │
            ▼
  Phase 3: SEC EDGAR &     ──► [GATE 3: Form 4, PEAD, FinBERT & 10-Q Diffs
           NLP Alt-Data                  Add Incremental IC > 0.02 Post-Cutoff]
            │
            ▼
  Phase 4: Calibration &   ──► [GATE 4: Brier Score Drops > 10% OOS;
           Kelly Sizing                  Zero-Skill Model Sizes to ~0.0%]
```

### Non-Negotiable Stop Rules
1. **No Modality Stacking Before Verification:** Do not build GAT, FinBERT, or Llama-3.1 SEC scrapers until Phase 1 passes Gate 1 on pure price/volume/microstructure data.
2. **Model Agnosticism at Gate 1:** If Sample-Weighted LightGBM/CatBoost outperforms the 2-layer Attention-LSTM/GRU on Phase 1–3 features out-of-sample, drop the sequence neural network or ensemble them with weights proportional to validation log-likelihood.
3. **Skepticism of Outsized Performance:** Any backtest producing an annualized Sharpe Ratio $> 2.5$ or win rate $> 65\%$ on 10-day swing horizons must be treated as a timestamp leak until proven otherwise.

---

## 3. Mathematical & Quantitative Specifications

### 3.1 Market- & Sector-Residualized Returns
To isolate stock-specific alpha from macro drift, compute rolling 60-day OLS betas against the market (`SPY`) and the stock's corresponding GICS Sector ETF ($\text{ETF}_s \in \{\text{XLK, XLV, XLY, XLP, XLE, XLI, XLB, XLRE, XLC}\}$), strictly using window $[t-60, t-1]$:
$$r_{i, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-1}}\right)$$
$$\epsilon_{i, t} = r_{i, t} - \left(\hat{\beta}_{i, \text{mkt}, t-1} \cdot r_{\text{SPY}, t} + \hat{\beta}_{i, \text{sec}, t-1} \cdot r_{\text{ETF}_s, t}\right)$$

### 3.2 Cross-Sectional Daily Rank Normalization
Financial feature distributions shift violently across bull and bear markets. To enforce cross-sectional stationarity on every trading day $t$, each raw feature $x_{i, t}$ across the $M_t \approx 250$ active universe stocks is mapped to a uniform rank in $[-1.0, +1.0]$:
$$\tilde{x}_{i, t} = 2 \cdot \left(\frac{\text{rank}(x_{i, t}) - 1}{M_t - 1}\right) - 1.0 \in [-1.0, +1.0]$$

### 3.3 Intraday-to-Daily Microstructure & Overnight Decomposition
Informed institutional repricing concentrates overnight and at the closing auction, while retail liquidity shocks occur intraday. Daily and hourly bars ($h = 1 \dots 7$) are decomposed into:
1. **Overnight Log-Return:**
   $$r_{\text{overnight}, t} = \ln\left(\frac{\text{Open}_t}{\text{Close}_{t-1}}\right)$$
2. **Intraday Session Log-Return:**
   $$r_{\text{intraday}, t} = \ln\left(\frac{\text{Close}_t}{\text{Open}_t}\right)$$
3. **Realized Intraday Volatility:**
   $$\sigma_{\text{intraday}, t} = \sqrt{\sum_{h=1}^7 r_{t, h}^2}$$
4. **Closing Power (Institutional EOD Auction Skew):**
   $$\text{CP}_t = \ln\left(\frac{V_{t, \text{hour 7}} + 10^{-5}}{V_{t, \text{hour 1}} + 10^{-5}}\right)$$
5. **Close-to-VWAP Deviation:**
   $$\Delta \text{VWAP}_t = \frac{P_{t, \text{close}} - \text{VWAP}_t}{\text{VWAP}_t}$$

### 3.4 Asynchronous Event-Decay Kernel & Feature Aging
Catalyst scores $X_0 \in [-1.0, 1.0]$ from quarterly filings, insider transactions, earnings shocks, and news are decayed continuously by trading days elapsed ($\Delta d$) since their point-in-time acceptance timestamp:
$$X_{\text{event}}(t) = X_0 \cdot \exp\left(-\frac{\ln(2)}{\tau} \cdot \Delta d\right)$$
* $\tau_{\text{news}} = 3.0\text{ trading days}$ (FinBERT headline sentiment momentum)
* $\tau_{\text{pead}} = 10.0\text{ trading days}$ (Post-Earnings Announcement Drift shock)
* $\tau_{\text{legal}} = 15.0\text{ trading days}$ (SEC 10-Q/10-K Item 1A & MD&A redline diffs)
* $\tau_{\text{form4}} = 20.0\text{ trading days}$ (SEC Form 4 open-market insider cluster buying/selling)

Each decayed feature is paired with an explicit normalized staleness age scalar:
$$\text{Age}_{\text{event}}(t) = \min\left(\frac{\Delta d}{63.0}, 1.0\right) \in [0.0, 1.0]$$

### 3.5 Variance-Sorted, Forward-Filtered 3-State Gaussian HMM
Fit a 3-state Gaussian HMM (with 10 random EM restarts) on standardized daily `SPY` log-returns and `VIX` levels:
1. **Forward Filtered Probabilities Only:** Compute forward variable $\alpha_t(k) = P(x_{1:t}, S_t = k)$ and normalize:
   $$p_t(k) = P(S_t = k \mid x_{1:t}) = \frac{\alpha_t(k)}{\sum_{j=0}^2 \alpha_t(j)}$$
   Backward-pass smoothed probabilities are strictly prohibited.
2. **State Ordering by Return Variance:** Eliminate label switching across walk-forward windows by sorting states in strictly ascending order of the market return variance component $\Sigma_{k, rr}$:
   $$\Sigma_{0, rr} < \Sigma_{1, rr} < \Sigma_{2, rr} \implies [\text{State 0: Low-Vol Bull, State 1: Sideways, State 2: High-Vol Crash}]$$

### 3.6 Residualized Triple-Barrier Labeling with Purging & Uniqueness Weights
Labels $y_{i, t} \in \{0: \text{Outperform/Buy}, 1: \text{Neutral/Hold}, 2: \text{Underperform/Sell}\}$ are generated on cumulative idiosyncratic residual returns $\sum_{\Delta=1}^{10} \epsilon_{i, t+\Delta}$ using dynamic volatility barriers:
* **Upper Barrier:** $+2.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* **Lower Barrier:** $-1.0 \cdot \sigma_{\text{ATR}(\epsilon), i, t}$
* **Vertical Barrier:** $H = 10\text{ trading days}$

For each sample $(i, t)$, record the exact barrier resolution date $t_{\text{end}} \le t + 10$:
* **Cross-Ticker Calendar-Date Purge:** For any training fold ending on calendar date $D_{\text{train}}$, purge all observations across all 250 tickers where $t_{\text{end}} > D_{\text{train}}$.
* **Sample-Uniqueness Weighting:** Weight each observation $(i, t)$ during training loss computation by its average inverse concurrency over its active lifespan:
  $$u_{i, t} = \left(\frac{1}{t_{\text{end}} - t + 1} \sum_{\tau = t}^{t_{\text{end}}} \frac{1}{c_{i, \tau}}\right), \quad c_{i, \tau} = \sum_{k} \mathbb{I}_{[t_{k, \text{start}}, t_{k, \text{end}}]}(\tau)$$

### 3.7 Signed Asymmetric Smart-Money Divergence
Decompose volume-weighted sentiment divergence $D_t = S_{\text{FinBERT}, t} \times \left(-\frac{\epsilon_t \cdot V_t}{\bar{V}_{20}}\right)$ into separate distribution and accumulation channels:
$$D_t^+ = \max(D_t, 0) \quad \text{(Media Euphoria into Institutional Selling / Distribution)}$$
$$D_t^- = \max(-D_t, 0) \quad \text{(Media Panic into Institutional Buying / Accumulation)}$$

### 3.8 Temperature Scaling & Multi-Outcome Kelly Sizing
1. **Post-Hoc Calibration:** Fit scalar temperature $T > 0$ via NLL minimization on an independent, purged validation split:
   $$\hat{p}_k = \frac{\exp(z_k / T)}{\sum_{j=0}^2 \exp(z_j / T)}$$
2. **Multi-Outcome Fractional Kelly Allocation:** Solve for optimal allocation $f^*$ maximizing expected log-wealth over ternary residual outcomes net of round-trip cost $c = 0.0015$:
   $$\max_{f \in [0, f_{\max}]} \mathbb{E}[\ln(1 + f \cdot R_{\text{net}})] = \hat{p}_{\text{buy}} \ln(1 + f (b_{\text{upper}} - c)) + \hat{p}_{\text{sell}} \ln(1 - f (|b_{\text{lower}}| + c)) + \hat{p}_{\text{hold}} \ln(1 + f (\bar{r}_{\text{vert}} - c))$$
   * Apply regime-dependent Fractional Kelly scalar $\kappa \in [0.25, 0.50]$: $f_{\text{alloc}} = \kappa \cdot f^*$.
   * Cap single-name allocation at $10\%$ of portfolio equity ($15\%$ absolute hard ceiling).

---

## 4. Complete Tensor Specifications & Feature Dictionary

All latent dimensions are right-sized to `hidden_size = 32` to prevent overfitting on daily financial sample sizes and run effortlessly on 24 GB Unified Memory.

```
===================================================================================================
                           COMPACT NEURAL ARCHITECTURE TOPOLOGY (v1.3)
===================================================================================================

 [Cross-Sectional Price & Flow]      [SEC EDGAR & NLP Alt-Data]     [Rolling Peer & GICS Graph]
     Shape: (Batch, 126, 15)              Shape: (Batch, 12)          Nodes: (Total_Nodes, 10)
                │                                  │                  Edges: (2, Total_Edges)
                ▼                                  ▼                             │
    ┌───────────────────────┐          ┌───────────────────────┐                 ▼
    │ 1-Layer GRU / LSTM    │          │ Linear(12 -> 32)      │       ┌───────────────────┐
    │ + 2-Head Self-Attn    │          │ + LayerNorm           │       │ 2-Layer PyG GAT   │
    │ hidden_size = 32      │          │ + GELU + Dropout(0.2) │       │ Root extracted    │
    └───────────┬───────────┘          └───────────┬───────────┘       │ via batch.ptr[:-1]│
                │ (B, 126, 32)                     │ (B, 1, 32)        └─────────┬─────────┘
                └────────────────┬─────────────────┘                             │
                                 ▼                                               │
                  ┌─────────────────────────────┐                                │
                  │ Cross-Modal Attention       │                                │
                  │ Q: Alt-Data (B, 1, 32)      │                                │
                  │ K, V: Temporal (B, 126, 32) │                                │
                  │ + Residual (t_126)          │                                │
                  └──────────────┬──────────────┘                                │
                                 │ (B, 32)                                       │ (B, 32)
                                 └──────────────────────┬────────────────────────┘
                                                        ▼
                                              Concat(32 + 32 + 2)
                                                  = (B, 66)
                                                        │
                                                        ▼
   [Filtered HMM Probabilities]               ┌───────────────────┐
       Shape: (Batch, 3) ───────────────────► │ Sigmoid Gate      │
                                              │ Linear(3 -> 66)   │
                                              └─────────┬─────────┘
                                                        │ (B, 66)
                                                        ▼
                                              ┌───────────────────┐
                                              │ Classification    │
                                              │ Linear(66 -> 32)  │
                                              │ + LayerNorm + SiLU│
                                              │ + Dropout(0.3)    │
                                              │ + Linear(32 -> 3) │
                                              └─────────┬─────────┘
                                                        │ Raw Logits: (B, 3)
                                                        ▼
                                              ┌───────────────────┐
                                              │ Temperature Scale │
                                              │ Softmax(z / T)    │
                                              └─────────┬─────────┘
                                                        │ Calibrated Probabilities: (B, 3)
                                                        ▼
                                              Multi-Outcome Kelly Sizer
```

### 4.1 Sequence Matrix Features: `(Batch, 126, 15)`
Every channel is cross-sectionally rank-normalized into $[-1.0, +1.0]$ across the 250-stock universe on each day $t$:

| Channel | Feature Name | Formula / Description | Quantitative Rationale |
| :---: | :--- | :--- | :--- |
| `0` | Raw Daily Log-Return | $r_t = \ln(\text{Close}_t / \text{Close}_{t-1})$ | Baseline total return |
| `1` | Idiosyncratic Residual Return | $\epsilon_t = r_t - (\hat{\beta}_{\text{mkt}} r_{\text{SPY},t} + \hat{\beta}_{\text{sec}} r_{\text{ETF},t})$ | Pure stock-specific return |
| `2` | Overnight Return | $\ln(\text{Open}_t / \text{Close}_{t-1})$ | Institutional news/earnings repricing |
| `3` | Intraday Session Return | $\ln(\text{Close}_t / \text{Open}_t)$ | Intraday liquidity & retail flow |
| `4` | Residual Momentum (5-day) | $\sum_{k=0}^4 \epsilon_{t-k}$ | Short-term idiosyncratic continuation |
| `5` | Residual Momentum (20-day) | $\sum_{k=0}^{19} \epsilon_{t-k}$ | Monthly idiosyncratic trend |
| `6` | Residual Momentum (60-day) | $\sum_{k=0}^{59} \epsilon_{t-k}$ | Quarterly post-catalyst drift |
| `7` | Relative Volume Ratio | $V_t / \bar{V}_{20}$ | Unusual activity participation |
| `8` | Amihud Illiquidity Z-Score | 60d Z-score of $\|r_t\| / (\text{Close}_t \cdot V_t)$ | Institutional price-impact absorption |
| `9` | ATR Volatility Ratio | $\text{ATR}_{14, t} / \text{Close}_t$ | Volatility regime scaling |
| `10` | Idiosyncratic Volatility (20d) | $\sqrt{252 \times \frac{1}{20}\sum_{k=0}^{19} \epsilon_{t-k}^2}$ | Residual risk penalty |
| `11` | Parkinson Range Volatility | $(\ln(H_t / L_t))^2 / (4 \ln 2)$ | Intraday high-low excursion variance |
| `12` | Realized Intraday Volatility | $\sqrt{\sum_{h=1}^7 r_{t, h}^2}$ from hourly bars | High-frequency noise vs. trend |
| `13` | Closing Power | $\ln((V_{\text{hour 7}} + 10^{-5}) / (V_{\text{hour 1}} + 10^{-5}))$ | Institutional closing auction dominance |
| `14` | Close-to-VWAP Deviation | $(P_{t, \text{close}} - \text{VWAP}_t) / \text{VWAP}_t$ | EOD accumulation above/below VWAP |

### 4.2 SEC EDGAR & NLP Alt-Data Vector: `(Batch, 12)`
| Index | Feature Name | Description | Default When Missing |
| :---: | :--- | :--- | :---: |
| `0` | Decayed FinBERT Sentiment | Headline sentiment decayed with $\tau = 3\text{d}$ half-life | `0.0` |
| `1` | News Staleness Age | $\min(\Delta d_{\text{news}} / 7.0, 1.0)$ | `1.0` |
| `2` | Decayed Form 4 Insider Score | Open-market cluster buying/selling (`P`/`S` codes, ex-10b5-1, $\tau = 20\text{d}$) | `0.0` |
| `3` | Form 4 Staleness Age | $\min(\Delta d_{\text{form4}} / 63.0, 1.0)$ | `1.0` |
| `4` | Decayed PEAD Earnings Shock | $\text{sign}(\epsilon_{\text{earn}}) \cdot \min(\|\epsilon_{\text{earn}}\| \cdot (V_{\text{earn}}/\bar{V}_{20}), 5.0)$ ($\tau = 10\text{d}$) | `0.0` |
| `5` | Earnings Staleness Age | $\min(\Delta d_{\text{earn}} / 63.0, 1.0)$ | `1.0` |
| `6` | Decayed 10-Q Legal Risk Delta | Item 1A/3 `difflib` redline risk scored by Llama-3.1 ($\tau = 15\text{d}$) | `0.0` |
| `7` | Decayed MD&A Catalyst Score | Item 2/7 MD&A redline operational tailwind score ($\tau = 15\text{d}$) | `0.0` |
| `8` | SEC Filing Staleness Age | $\min(\Delta d_{\text{filing}} / 63.0, 1.0)$ | `1.0` |
| `9` | Missing Mask: News | Binary indicator: `1.0` if news in last 7d, `0.0` otherwise | `0.0` |
| `10` | Missing Mask: Form 4 | Binary indicator: `1.0` if discretionary Form 4 in last 63d, `0.0` otherwise | `0.0` |
| `11` | Missing Mask: 10-Q/10-K | Binary indicator: `1.0` if valid SEC redline diff in last 63d, `0.0` otherwise | `0.0` |

### 4.3 Dynamic Rolling Peer & GICS Graph Matrix: `(Total_Nodes, 10)`
To eliminate look-ahead survivorship bias from manual supply-chain curation, graphs are constructed algorithmically at the start of each walk-forward month using trailing 60-day data:
* **Node $v_0$:** Target Company.
* **Nodes $v_1 \dots v_4$:** Top 4 GICS Sub-Industry peers by market cap similarity.
* **Nodes $v_5 \dots v_8$:** Top 4 universe stocks with highest 5-day lead-lag cross-correlation ($\text{corr}(r_{j, t-5:t-1}, r_{0, t})$).
* **Node $v_9$:** Corresponding GICS Sector ETF.

All fundamental features are computed strictly from the **SEC EDGAR XBRL `companyfacts` API stamped by the `filed` date** (never back-dated to fiscal period end):
| Index | Node Feature Description |
| :---: | :--- |
| `0` | 20-day residual return $\sum \epsilon_{t-19:t}$ |
| `1` | 5-day residual return $\sum \epsilon_{t-4:t}$ |
| `2` | 60-day rolling Beta vs SPY ETF |
| `3` | Point-in-Time Earnings Yield ($\text{TTM EPS}_{\text{filed}} / \text{Price}_t$) peer Z-score |
| `4` | Point-in-Time YoY Quarterly Revenue Growth ($\text{Rev}_{\text{filed}}$) peer Z-score |
| `5` | Point-in-Time Operating Margin ($\text{OpInc}_{\text{filed}} / \text{Rev}_{\text{filed}}$) peer Z-score |
| `6` | 20-day Realized Idiosyncratic Volatility peer Z-score |
| `7` | 5-day Lead-Lag cross-correlation coefficient with Target Node ($v_0$) |
| `8` | Normalized Market Cap: $\ln(\text{MarketCap}_t) / 10$ |
| `9` | Categorical Node Role: $0.0 = \text{Target}, 0.33 = \text{GICS Peer}, 0.66 = \text{Lead-Lag Peer}, 1.0 = \text{Sector ETF}$ |

*Batch Extraction:* Root target node $v_0$ for every graph in the mini-batch is extracted via `h_target = g_out[batch.ptr[:-1]]`, yielding `(Batch, 32)`.

### 4.4 Fusion, Regime Gate & Classification Head
* **Signed Divergence Vector:** $[D_t^+, D_t^-] \in \mathbb{R}^{(B, 2)}$
* **Concatenated Latent Representation:**
  $$\mathbf{h}_{\text{fused}} = [\mathbf{h}_{\text{cross\_modal}}^{(32)} \,\|\, \mathbf{h}_{\text{graph}}^{(32)} \,\|\, \mathbf{D}_t^{(2)}] \in \mathbb{R}^{(B, 66)}$$
* **Filtered Macro Regime Gate:**
  $$\mathbf{g} = \sigma(\text{Linear}_{3 \to 66}(\mathbf{p}_{\text{filtered}})) \in (0, 1)^{(B, 66)}$$
  $$\mathbf{h}_{\text{gated}} = \mathbf{h}_{\text{fused}} \odot \mathbf{g} \in \mathbb{R}^{(B, 66)}$$
* **Classification Head (`LayerNorm` Enforced):**
  $$\mathbf{z} = \text{Linear}_{32 \to 3}(\text{Dropout}_{0.3}(\text{SiLU}(\text{LayerNorm}(\text{Linear}_{66 \to 32}(\mathbf{h}_{\text{gated}}))))) \in \mathbb{R}^{(B, 3)}$$
  *Note:* `LayerNorm` replaces `BatchNorm1d` to prevent running-statistic drift across non-stationary test windows. Softmax is deferred strictly to `TemperatureScaledCalibrator`.

---

## 5. Point-in-Time Data Sources & Universe Definition

### 5.1 Asset Universe Selection
* **Constituents:** 250 liquid US mid-to-large-cap stocks selected from S&P 500 (excluding top-50 mega-cap outliers where optional, or S&P 500 Rank 100–500 + S&P 400 MidCap) with 90-day median daily dollar volume $> \$25\text{M}$.
* **Exclusions:** Financials (GICS 40) and Utilities (GICS 55).
* **Survivorship Mitigation:** Universe constituents are snapshotted annually at the start of each walk-forward year using historical index/liquidity tables.

### 5.2 Primary Data Feeds & PIT Rules
| Domain | Primary Source | Fallback Source | Point-in-Time Stamping Rule |
| :--- | :--- | :--- | :--- |
| **Daily/Hourly OHLCV** | Polygon.io Stocks API | `yfinance` / Alpaca | Hourly bars aggregated to 16:00 EST close; split/dividend adjusted |
| **SEC Form 4 Insiders** | SEC EDGAR XML API | OpenInsider / Finnhub | Stamped by exact `acceptanceDateTime`; $>16:00\text{ EST} \to t+1$ |
| **SEC 10-Q/10-K Text** | SEC EDGAR Submission API | `sec-edgar-downloader` | Stamped by `acceptanceDateTime`; $>16:00\text{ EST} \to t+1$ |
| **PIT Fundamentals** | SEC EDGAR XBRL `companyfacts` | Polygon Financials (by `filing_date`) | Keyed strictly on `filed` date; fiscal period `end` date forbidden |
| **News Headlines** | Finnhub / Polygon News | Alpaca Benzinga News | Articles published $> 16:00\text{ EST}$ shifted to $t+1$ |

---

## 6. Phased Implementation Roadmap & Falsification Gates

### Phase 0: Lock Spec v1.3 (Complete)
* All mathematical equations, residualized targets, PIT rules, and compact `(32)` tensor shapes are locked with zero unresolved parameters.

### Phase 1: Residualized Leakage-Safe Backbone (Weeks 1–2)
* Build `market_loader.py` to ingest daily/hourly OHLCV for the 250-stock universe plus `SPY` and the 9 GICS Sector ETFs.
* Build `features.py` computing the 15 cross-sectionally rank-normalized channels (overnight/intraday split, residual momentum, Amihud illiquidity, Closing Power, Close-to-VWAP).
* Build `triple_barrier.py` on residualized returns $\epsilon_{i,t}$ storing exact $t_{\text{end}}$ timestamps.
* Build `walk_forward.py` enforcing cross-ticker calendar-date purging and sample-uniqueness weights $u_{i,t}$.
* Train and compare:
  1. **Sample-Weighted LightGBM / CatBoost** on flattened rolling summary features.
  2. **Compact 1-Layer GRU/LSTM + Self-Attention (`hidden_size=32`)**.
* **Gate 1 Criteria:**
  1. Shuffled-label test yields Sharpe $\approx 0.0 \pm 0.15$ and ROC-AUC $\approx 0.50$.
  2. 1-day forward-shift sanity test confirms pipeline explodes to Sharpe $> 4.0$ when deliberately leaked, and drops to realistic values when purged.
  3. Out-of-sample return net of 15 bps round-trip friction beats equal-weight universe baseline and achieves Information Coefficient $IC > 0.025$.

### Phase 2: Forward-Filtered HMM & Dynamic Peer GAT (Weeks 3–4)
* Build `regime_hmm.py` with 10 random EM restarts, forward-filtered probabilities $P(S_t \mid x_{1:t})$, and return-variance state sorting.
* Build `graph_builder.py` generating monthly rolling GICS + Lead-Lag peer graphs with SEC XBRL `filed`-date fundamentals and `batch.ptr[:-1]` root node slicing.
* Run PyG GAT strictly on CPU.
* **Gate 2 Criteria:**
  * Ablation confirms the HMM gate reduces maximum out-of-sample portfolio drawdown by $\ge 15\%$, and the Peer GAT branch improves out-of-sample log-likelihood / Sharpe over Phase 1 alone (otherwise prune GAT and retain HMM).

### Phase 3: SEC EDGAR Catalysts (Form 4, PEAD, 10-Q Redlines & FinBERT) (Weeks 5–6)
* Build `form4_insider.py` parsing SEC EDGAR Form 4 XMLs for open-market purchases/sales (`P`/`S` codes, filtering out Rule 10b5-1 scheduled trades).
* Build `sec_diff.py` running `difflib` on consecutive 10-Q/10-K Item 1A & MD&A sections, scored via local Ollama (`Llama-3.1-8B-Instruct` Q8_0/Q4_K_M, temperature 0, strict JSON schema).
* Build `news_sentiment.py` using local `ProsusAI/finbert` with 512-token chunking and $\tau = 3\text{d}$ decay.
* Evaluate Llama-3.1 legal/MD&A features **strictly on post-cutoff dates** (post-mid-2024) to prevent pre-training look-ahead memorization.
* **Gate 3 Criteria:**
  * Ablation confirms Alt-Data features provide statistically significant incremental IC ($\Delta IC > +0.015$, $t\text{-stat} > 2.0$) on the post-cutoff evaluation window.

### Phase 4: Temperature Calibration & Multi-Outcome Kelly Engine (Weeks 7–8)
* Build `calibration.py` fitting `TemperatureScaledCalibrator` on a dedicated, purged validation fold.
* Build `kelly_engine.py` solving the multi-outcome expected log-wealth objective net of 15 bps friction, scaled by regime fraction $\kappa \in [0.25, 0.50]$ and capped at $10\%$ single-stock weight.
* Run full portfolio simulation in `vectorbt`.
* **Gate 4 Criteria:**
  1. Out-of-sample Brier score improves by $> 10\%$ after temperature scaling.
  2. Feeding uninformative random logits into `kelly_engine.py` results in $0.0\%$ allocation across all stocks.
  3. Portfolio maximum drawdown remains $< 15\%$ across out-of-sample walk-forward windows.

---

## 7. M5 Air (24 GB) Compute & Runtime Constraints

### 7.1 Unified Memory Allocation Budget (24 GB)
```
Total Unified Memory: 24,576 MB
├── macOS & System Core:           ~4,600 MB (Constant)
├── PyTorch / LightGBM Workspace:  ~2,800 MB (Reduced via hidden_size=32)
└── Available Headroom:           ~17,176 MB
```
* **Ollama (`Llama-3.1-8B` Q8_0 / Q4_K_M):** ~5,000–8,500 MB (Runs offline as a sequential batch job).
* **FinBERT Batch Scorer:** ~1,500 MB.
* **Process Isolation Rule:** Never run Ollama inference and model training concurrently. Pre-compute all SEC Form 4, 10-Q diff scores, and FinBERT embeddings into local **Apache Parquet** feature stores (`data/features/*.parquet`). Model training reads only pre-computed numeric Parquet files.

### 7.2 Thermal Throttling Mitigation
* **Fanless Envelope:** Sustained full-load compute throttles M5 package power after ~12–15 minutes.
* **Execution Rules:**
  1. With `hidden_size=32` and pre-computed Parquet caches, a full walk-forward epoch over 250 stocks completes in seconds rather than minutes, keeping the M5 Air inside its burst thermal envelope without throttling.
  2. Initial historical 10-Q LLM diff extraction runs once as a checkpointed overnight batch job.

---

## 8. Target Output API Contract (JSON Schema)

```json
{
  "ticker": "CRWD",
  "timestamp_utc": "2026-10-19T20:00:00Z",
  "horizon_days": 10,
  "winning_engine": "ENSEMBLE_LGBM_GRU32",
  "data_integrity": {
    "point_in_time_shifted": true,
    "xbrl_filed_date_verified": true,
    "missing_modalities": []
  },
  "macro_regime": {
    "current_state": "LOW_VOL_BULL",
    "state_index": 0,
    "filtered_probabilities": {
      "bull": 0.812,
      "sideways": 0.145,
      "crash": 0.043
    }
  },
  "signal_breakdown": {
    "residual_momentum_20d_rank": 0.78,
    "overnight_vs_intraday_skew": 0.42,
    "amihud_illiquidity_z": -0.31,
    "closing_power_z": 1.15,
    "decayed_finbert_sentiment": 0.54,
    "decayed_form4_insider_cluster": 0.82,
    "decayed_pead_shock": 0.35,
    "decayed_10q_legal_risk_delta": -0.04,
    "peer_gat_spillover": 0.49,
    "smart_money_divergence_pos": 0.0,
    "smart_money_divergence_neg": 0.19
  },
  "calibration": {
    "temperature": 1.34,
    "calibrated_probabilities": {
      "outperform_buy": 0.635,
      "neutral_hold": 0.245,
      "underperform_sell": 0.120
    }
  },
  "execution_recommendation": {
    "action": "BUY",
    "conviction_tier": "HIGH",
    "recommended_portfolio_allocation_pct": 6.40,
    "entry_price": 312.40,
    "dynamic_take_profit": 334.80,
    "dynamic_stop_loss": 301.20,
    "reward_to_risk_ratio": 2.0,
    "estimated_round_trip_cost_bps": 15.0
  }
}
```
