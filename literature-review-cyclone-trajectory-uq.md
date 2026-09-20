# Literature Review: Cyclone Trajectory Modeling, Probability & Uncertainty Quantification

**Context.** Hypothesis: correlations (spatial/temporal dependence structure) evolve as a cyclone propagates. Goal: a model that learns trajectories, computes probabilities over future positions, and quantifies how much a track can deviate — treating the cyclone as a spatio-temporal process.

**Date compiled:** 2026-07-20

---

## 1. Operational baselines (what forecasters actually use)

These are the benchmarks any new model must beat, and the simplest formalization of "how much can a track deviate."

- **NHC cone of uncertainty** — track uncertainty as the 67th percentile of *historical* track-error distributions. Static, not situation-dependent. [Guide to probabilistic wind hazard](https://hurricanes.ral.ucar.edu/guide/probwind/index.php)
- **Monte Carlo Wind Speed Probability (WSP) model** — 1,000 track realizations sampled from 5-year official error distributions; yields wind-speed exceedance probabilities. [DeMaria et al., NHC](https://www.nhc.noaa.gov/pdf/2009waf_wsp.pdf); [threshold selection, WAF 2014](https://journals.ametsoc.org/view/journals/wefo/29/5/waf-d-13-00100_1.xml)
- **GPCE (Goerss Predicted Consensus Error)** — regression on ensemble *spread* to predict situation-dependent consensus track error; added to WSP in 2011. [CIRA overview](https://rammb.cira.colostate.edu/research/tropical_cyclones/tc_wind_prob/)

**Takeaway:** operational UQ is largely climatological error statistics + Monte Carlo. The literature gap your project targets — flow/state-dependent, learned uncertainty — is real.

## 2. Stochastic / statistical track models (synthetic-track tradition)

The classic answer to "model the trajectory as a stochastic process." Used heavily in risk assessment (insurance, building codes).

- **Vickery et al.** — empirical autoregressive track model over the Atlantic; first fully statistical synthetic-track approach. [Empirical track model](https://www.researchgate.net/publication/245304500_Simulation_of_Hurricane_Risk_in_the_US_Using_Empirical_Track_Model)
- **Emanuel et al. (2006)** — coupled statistical track generation with deterministic intensity model (statistical–dynamical hybrid).
- **Markov / AR track models** — translation speed and heading autocorrelation spectra justify Markov-process modeling; explicit treatment of autocorrelation in track shape and cyclone lysis (termination). [Hall & Jewson, track-shape autocorrelation](https://arxiv.org/pdf/physics/0509024); [cyclone lysis](https://arxiv.org/pdf/physics/0512091)
- **STORM dataset** — global synthetic TC hazard set from statistical resampling. [Bloemendaal et al., Sci Data 2020](https://www.nature.com/articles/s41597-020-0381-2)
- **PepC (Princeton)** — hierarchical Poisson genesis + analog-wind track + Markov intensity. [statistical–parametric model, NHESS 2021](https://nhess.copernicus.org/articles/21/893/2021/)
- **Multivariate functional PCA** — simulates whole tracks (lat, lon, intensity as functions of time) "at one go," capturing along-track correlation globally rather than step-by-step. [Yang et al., 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EA001748)

**Relevance:** these define the probabilistic vocabulary (genesis → propagation → lysis; AR coefficients as local dynamics) and are directly extensible: your "evolving correlation" hypothesis = making the AR/Markov transition kernel state- and location-dependent.

## 3. Bayesian & Gaussian-process approaches (explicit UQ)

- **Bayesian multivariate functional models** — lat/lon/wind as time-functions with spatially varying coefficients; full posterior via MCMC. [Spatial Statistics](https://www.sciencedirect.com/science/article/abs/pii/S2211675317302841)
- **Bayesian hierarchical models for TC forecast-error fields** — forecast errors modeled as Gaussian processes; hierarchical priors on spatial-correlation parameters — the closest existing formalization of "learning how correlation evolves." [arXiv 2210.16683](https://arxiv.org/pdf/2210.16683)
- **Marked point processes** for seasonal cyclone occurrence with GP intensities. [arXiv 1506.00429](https://arxiv.org/pdf/1506.00429)
- **Bayesian neural networks** for multi-step cyclone intensity with predictive distributions. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-29911-8_22)
- **GP emulators** for spatio-temporal storm surge given track parameters. [ResearchGate](https://www.researchgate.net/publication/364469335_Spatio-temporal_storm_surge_emulation_using_Gaussian_Process_techniques)

## 4. Deep learning trajectory models (deterministic → probabilistic)

- **RNN track prediction** on HURDAT. [Alemany et al., AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/3819/3697)
- **Hurricast** — multimodal: encoder–decoder features from reanalysis maps + XGBoost; track + intensity. [WAF 2022](https://journals.ametsoc.org/view/journals/wefo/37/6/WAF-D-21-0091.1.xml)
- **GAN-based**: typhoon track GAN from satellite + met data ([arXiv 1812.01943](https://arxiv.org/pdf/1812.01943)); **MGTCF** — multi-generator GAN with environment net and generator-selection net (multi-modal futures ≈ implicit distribution over tracks).
- **GraphTransformers** on HURDAT — explicit graph structure over geospatial sequence improves 6-hourly track prediction. [arXiv 2310.20174](https://arxiv.org/html/2310.20174)
- **Temporal Fusion Transformer** — quantile forecasts give native UQ for intensity, interpretable attention. [Sci Reports 2025](https://www.nature.com/articles/s41598-025-15522-7)

## 5. Learned uncertainty & calibration (most directly on-target)

- **Probabilistic NN predicting bivariate normal track-error distributions** — DeMaria et al. 2025: NN outputs a 2D Gaussian per lead time; better calibrated than NHC's current cone methodology. This is essentially the minimal version of your project. [AIES 2025](https://journals.ametsoc.org/view/journals/aies/4/2/AIES-D-24-0066.1.xml) / [arXiv](https://arxiv.org/html/2503.09840v1)
- **Conformal prediction for TC tracks** — distribution-free, finite-sample-valid uncertainty sets around track forecasts. [ESWA 2024](https://www.sciencedirect.com/science/article/abs/pii/S0957417424006092)
- **Situation-dependent track uncertainty via RNN**. [Nat Hazards 2026](https://link.springer.com/article/10.1007/s11069-026-08023-x)
- **PTCIF** — probabilistic deep learning over multimodal spatio-temporal data for TC forecasting.

## 6. Generative / diffusion ensembles (state of the art for spatio-temporal probability)

- **GenCast (DeepMind)** — diffusion model sampling from the *joint* distribution of weather across space and time; beat ECMWF ENS on TC track across 1–5 day lead times, well-calibrated. The strongest existing demonstration that a learned generative model captures spatio-temporally correlated forecast uncertainty. [Science Advances (ensemble emulation with diffusion)](https://www.science.org/doi/10.1126/sciadv.adk4489)
- **Cascaded diffusion** for cyclone forecast + super-resolution + precipitation from satellite data. [arXiv 2310.01690](https://arxiv.org/html/2310.01690v6)
- **Phys-Diff** — physics-inspired latent diffusion; N=50 samples from noise → ensemble mean + UQ. [arXiv 2603.00521](https://arxiv.org/html/2603.00521)
- **TCDM** — conditional diffusion generating full probability distribution of intensity from multimodal inputs. [Remote Sensing 2025](https://www.mdpi.com/2072-4292/17/21/3600)
- **Learnable-perturbation ensembles** and **fast physics-based perturbation generators** for ML weather models — cheap ensemble spread for TC tracks. [npj Clim Atmos Sci](https://www.nature.com/articles/s41612-025-01009-9); [arXiv 2510.23794](https://arxiv.org/html/2510.23794)
- AI-NWP context: Pangu-Weather, GraphCast, FuXi, FengWu track-error comparisons; operational adoption (HKO, 2025 season). [npj 2024 evaluation](https://www.nature.com/articles/s41612-024-00769-0)

## 7. Evolving correlation structure (your specific hypothesis)

Directly relevant but sparser literature:

- **Wind-speed residual correlation varies per TC** — the spatial correlation structure of wind-field residuals is storm-specific, and ignoring it biases risk estimates. Direct evidence that correlation is dynamic, not fixed. [Spatial correlation & wind speed uncertainties](https://www.researchgate.net/publication/233752160_Spatial_Correlation_and_Wind_Speed_Uncertainties_of_Hurricane_Wind_Field_Model)
- **Spatio-temporal extremes** — max-stable processes, Bayesian hierarchical models, copulas for storm fields. [Spatio-temporal modelling of extreme storms](https://arxiv.org/pdf/1501.06377)
- **Time-varying copulas** — spatio-temporal copulas as time-indexed mixtures of spatial copulas; changepoint–copula frameworks capture regime shifts in TC dependence structure. [Stats 2026](https://doi.org/10.3390/stats9030059)

---

## Synthesis: positioning your model

**The gap.** Almost all track-UQ work predicts *marginal* uncertainty per lead time (a Gaussian or quantiles at t+6h, t+12h, …). Very little learns the *joint, along-track correlation structure* and how it evolves with the storm state — yet that joint structure is exactly what determines realistic deviation envelopes (a storm that deviates left at 24h stays left-biased at 48h).

**A defensible framing:**
1. **State space.** Track as a stochastic process X(t) = (lat, lon, [intensity]); condition transition dynamics on environmental fields (steering flow, SST, shear) — the statistical–dynamical tradition (§2).
2. **Learn the transition kernel** with a neural model (transformer/GNN over the spatio-temporal neighborhood, §4) but output a *distribution*, not a point: mixture density, bivariate Gaussian (§5), or a conditional diffusion sampler (§6).
3. **Correlation evolution** enters as the model's predicted covariance Σ(t | state) — test your hypothesis by checking whether learned Σ varies systematically with storm phase (genesis, recurvature, extratropical transition) vs. a static climatological Σ (the NHC-cone null hypothesis).
4. **Calibrate and validate** with conformal prediction (§5) on held-out storms; verify with CRPS, spread–skill ratio, and rank histograms against the NHC cone and GPCE baselines (§1).

**Datasets:** HURDAT2 / IBTrACS (best tracks), ERA5 (environmental fields), NHC forecast-error archives (for §5-style error modeling), TIGGE/ECMWF ENS (ensemble baselines).

**Nearest prior work to beat/extend:** DeMaria et al. 2025 (bivariate-normal PNN — marginal only), GenCast (joint but full-atmosphere, expensive), functional-PCA synthetic tracks (joint correlation but climatological, not state-dependent).
