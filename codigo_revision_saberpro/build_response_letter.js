const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
} = require("docx");
const fs = require("fs");

const PAGE = { size: { width: 12240, height: 15840 } }; // US Letter

function heading(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 300, after: 150 } });
}

function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 200 },
    alignment: AlignmentType.JUSTIFIED,
  });
}

function commentPara(text) {
  return new Paragraph({
    children: [new TextRun({ text: `Reviewer comment: "${text}"`, italics: true })],
    spacing: { before: 200, after: 120 },
    alignment: AlignmentType.JUSTIFIED,
  });
}

function responseLabel() {
  return new Paragraph({
    children: [new TextRun({ text: "Response:", bold: true })],
    spacing: { after: 80 },
  });
}

function pending(idLabel) {
  return new Paragraph({
    children: [new TextRun({
      text: `[PENDIENTE — ${idLabel} aún no se ha resuelto en este borrador]`,
      italics: true, color: "999999",
    })],
    spacing: { before: 100, after: 300 },
  });
}

const children = [];

// ---------- Title ----------
children.push(new Paragraph({
  children: [new TextRun({ text: "Response to Reviewers", bold: true, size: 32 })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 100 },
}));
children.push(new Paragraph({
  children: [new TextRun({
    text: "Structural Inequality in Higher Education Performance: Unsupervised Profiling of "
      + "452,020 Colombian Saber Pro Students Using UMAP and K-Means",
    italics: true, size: 24,
  })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 100 },
}));
children.push(new Paragraph({
  children: [new TextRun({ text: "Large-scale Assessments in Education", size: 22 })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 400 },
}));

children.push(body(
  "We thank the Editor and both reviewers for their thorough and constructive evaluation of "
  + "our manuscript. In response to their comments, we carried out a substantial set of "
  + "additional analyses, which we believe have strengthened the manuscript considerably — "
  + "while also requiring us to temper several of our original claims in the interest of "
  + "accuracy. Below we address each comment individually. Reviewer and Editor comments are "
  + "reproduced in italics; our responses follow, together with a description of the "
  + "corresponding changes made to the manuscript."
));

// ================= EDITOR =================
children.push(heading("Response to the Editor", HeadingLevel.HEADING_1));

children.push(heading("Editor comment 1 (justification and documentation of K = 8)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "There is a need for stronger justification and reproducibility of the clustering "
  + "solution, including the selection of eight clusters. Along these lines, clearer "
  + "documentation of the analytic settings should be provided."
));
children.push(responseLabel());
children.push(body(
  "We agree that the original manuscript did not present the quantitative basis for K = 8 "
  + "with sufficient clarity. We re-examined the complete Phase 1 model-selection grid "
  + "(28 KMeans and 28 GMM configurations spanning K = 2-8 and UMAP dimensionality 2-5, plus "
  + "DBSCAN and HDBSCAN sweeps). K = 8 is not the single optimum on any one internal index: "
  + "the highest Silhouette value among KMeans solutions occurs at K = 2 (0.460), and the "
  + "lowest (best) Davies-Bouldin value occurs at K = 4 (0.825). However, K = 8 ranks second "
  + "on Davies-Bouldin among all 28 KMeans configurations (0.842, within 5% of the best value) "
  + "and remains competitive on Silhouette (0.409). DBSCAN configurations that achieve higher "
  + "raw Silhouette values (up to 0.83) do so only by labelling 83-99% of observations as noise, "
  + "which we already characterise in the manuscript as operationally unusable for institutional "
  + "decision-making. We have added a new supplementary figure reporting the elbow (inertia), "
  + "Silhouette and Davies-Bouldin curves for K = 2-8 side by side (new Figure S1), and we now "
  + "state explicitly, rather than implicitly, that K = 8 was chosen by balancing near-optimal "
  + "internal validity (Davies-Bouldin) against the interpretability required for the study's "
  + "equity-oriented aims: coarser solutions (K = 2-4) collapse distinctions between subgroups "
  + "that are substantively important for the paper's argument (e.g., the contrast between the "
  + "profiles combining similar academic performance with different structural constraints). We "
  + "revised Section 4.2 to present this reasoning explicitly as a trade-off rather than "
  + "implying that K = 8 is uniquely optimal by internal indices alone. As additional, "
  + "independent support, our response to Reviewer 2's third comment below reports that "
  + "HDBSCAN — run with a min_cluster_size appropriate to this sample size, rather than the "
  + "very small values used in the original Phase 1 sweep — converges on its own to exactly "
  + "eight clusters, with very low noise, consistently across four UMAP dimensionalities. We "
  + "view this convergence of a density-based method that does not presuppose K as meaningful "
  + "corroboration of the eight-cluster solution, beyond the internal-index comparison above."
));

children.push(heading("Editor comment 2 (stability of the complete UMAP + K-Means pipeline)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Stronger evidence of stability should be provided regarding the complete UMAP-K-means "
  + "procedure, as opposed to just the effects of K-means initialization while holding the "
  + "UMAP representation fixed."
));
children.push(responseLabel());
children.push(body(
  "We implemented exactly the test the Editor and Reviewer 2 (comment 2) requested. We reran "
  + "the complete pipeline 15 times, varying in every repetition both the UMAP random seed and "
  + "the 80,000-observation fitting sample (drawn independently each time), and evaluated the "
  + "resulting K = 8 cluster assignments on a fixed 50,000-observation evaluation set. The mean "
  + "pairwise Adjusted Rand Index (ARI) across the 15 repetitions is 0.506 (SD = 0.085, "
  + "range 0.328-0.728) — notably lower than the 0.69 originally reported, which varied only "
  + "the K-Means initialization while holding the UMAP embedding fixed. This result was "
  + "independently replicated by the corresponding author on a separate computing environment "
  + "(mean ARI = 0.501, SD = 0.075), which we take as evidence that the finding is a genuine "
  + "property of the pipeline rather than an artefact of a particular hardware or software "
  + "configuration. We also examined stability at the cluster level: agreement with a reference "
  + "labelling (via Hungarian alignment) ranges from approximately 0.32-0.36 for the least "
  + "stable clusters to 0.72-0.88 for the most stable ones. We have added the complete-pipeline "
  + "ARI distribution and a consensus/co-clustering matrix as a new supplementary figure, and we "
  + "revised the Results and Discussion to report this more conservative ARI as the primary "
  + "stability measure, explicitly distinguishing profiles with strong replication support from "
  + "those that should be treated as more tentative."
));

children.push(heading("Editor comment 3 (relationship between profiles, validation, and applications)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "The relationship between the descriptive profiles, their validation, and their proposed "
  + "applications should be clarified. In particular, it will be important to talk further "
  + "about how particular profile characteristics could inform equity-oriented institutional "
  + "decisions."
));
children.push(responseLabel());
children.push(body(
  "We agree that the link between the descriptive profiles, their validation, and their "
  + "proposed applications needed to be made explicit, and we have substantially revised the "
  + "Discussion and Conclusions accordingly. Rather than presenting the eight profiles as an "
  + "undifferentiated list, we now organize the discussion around concrete, data-grounded "
  + "contrasts between profiles that share similar starting conditions but differ in the "
  + "constraints they face, since these contrasts are what make the typology actionable for "
  + "equity-oriented decisions. Two contrasts illustrate the framework. First, Cluster 4 "
  + "(18.3% of students; heavy paid work — mean 3.8 on our 0-4 hours-worked scale, the highest "
  + "of any cluster — overwhelmingly self-financed, 83.1%, and the lowest parental financial "
  + "support, 13.2%) has a mean global score of 138.0, while Cluster 7 (14.5% of students; "
  + "lower socioeconomic stratum than Cluster 4 and lower internet access, 85.3% versus 93.8%, "
  + "but low work hours, 1.9, and the highest scholarship coverage of any cluster, 49.7%) "
  + "achieves a higher mean score of 146.4 despite starting from a less advantaged position on "
  + "stratum and connectivity. Second, Cluster 0 (15.8% of students; the lowest stratum, 1.8, "
  + "the lowest internet access, 78.0%, and largely self-financed, 56.7%, despite comparatively "
  + "high scholarship coverage, 35.9%) has the lowest mean score among the seven "
  + "typical-sized profiles, 135.3, combining several structural constraints at once. We present "
  + "these as hypothesis-generating patterns, not causal claims: the cross-sectional design "
  + "cannot establish that reducing work hours or expanding scholarship coverage would raise "
  + "scores for a given student, only that students who already combine these characteristics "
  + "score differently on average. We state this limitation explicitly wherever the framework "
  + "is discussed. On how profile characteristics could inform equity-oriented institutional "
  + "decisions, we now propose a concrete framework: for each profile we report its defining "
  + "combination of stratum, work hours, financing source, and connectivity (new Figure S_profile), "
  + "and we discuss, for illustration, what kind of institutional response each combination "
  + "points to as a hypothesis worth piloting — for example, flexible scheduling or work-study "
  + "arrangements for the Cluster-4 pattern (heavy paid work, self-financed, low parental "
  + "support), connectivity subsidies for the Cluster-0 pattern (low stratum, low internet "
  + "access), and continued or expanded scholarship coverage for the Cluster-7 pattern, given "
  + "its more favourable outcomes relative to Cluster 4 despite a less advantaged starting "
  + "position. Regarding what variables or patterns should trigger a specific decision in "
  + "practice: per our response to Reviewer 2's sixth comment above, cluster membership does "
  + "not carry predictive information beyond a combination of the underlying covariates "
  + "(socioeconomic stratum, hours worked, financing source, connectivity), which institutions "
  + "already collect at admission or enrollment. We therefore clarify that the operational "
  + "trigger for an intervention should be these underlying, directly observable variables "
  + "themselves, not cluster membership computed from a UMAP + K-Means pipeline; the eight-"
  + "profile typology's role is to organize and communicate how these variables co-occur in "
  + "practice, in an interpretable form suitable for institutional decision-makers, not to "
  + "serve as a required computational step. We have also added a paragraph connecting this "
  + "framework to the robustness evidence reported elsewhere in this letter: the profiles are "
  + "reasonably stable across independent pipeline reruns (Editor comment 2), reproduce "
  + "independently across four examination administrations (Reviewer 2, comment 8), and are not "
  + "artefacts of any single institution or programme (Reviewer 2, comment 10), which together "
  + "support treating them as a reasonably reliable descriptive basis for the proposed "
  + "framework — while the calibration analysis (Reviewer 2, comment 1) and the moderate "
  + "complete-pipeline stability (Editor comment 2) are reasons for the hypothesis-generating "
  + "framing we now use throughout, rather than presenting the framework as validated."
));

// ================= REVIEWER 1 =================
children.push(heading("Response to Reviewer 1", HeadingLevel.HEADING_1));

children.push(heading("Reviewer 1, comment 1 (operational detail on UMAP and clustering)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "It would be desirable to expand (or supplement) the operational details, particularly "
  + "regarding UMAP hyperparameters, clustering initialization and consistency criteria, and "
  + "how process variability is controlled (e.g., random seeds, clustering sensitivity, and "
  + "stability across runs)."
));
children.push(responseLabel());
children.push(body(
  "We have added a supplementary table reporting the complete UMAP configuration used "
  + "throughout the pipeline (n_neighbors = 10, random state, 80,000-observation fitting "
  + "sample, Euclidean metric, target dimensionality) and the K-Means initialization procedure "
  + "(MiniBatchKMeans, batch size 10,000, automatic n_init). Regarding process variability and "
  + "stability across runs specifically, please see our response to the Editor's second comment "
  + "above, which reports the complete-pipeline stability analysis (15 repetitions varying both "
  + "the UMAP seed and the fitting sample) requested here and by Reviewer 2."
));

children.push(heading("Reviewer 1, comment 2 (empirical support for K = 8)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Although the selection of K = 8 is mentioned, the empirical support for this choice could "
  + "be presented more clearly, including the criteria used (e.g., the elbow method, "
  + "Davies-Bouldin index, or other cluster validity measures)."
));
children.push(responseLabel());
children.push(body(
  "Please see our response to the Editor's first comment above, where we report the complete "
  + "elbow, Silhouette and Davies-Bouldin comparison across K = 2-8 (new Figure S1) and state "
  + "explicitly the basis on which K = 8 was chosen."
));

children.push(heading("Reviewer 1, comment 3 (from profiles to intervention plans)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Given that the objective is “equity-oriented decision-making,” it would be valuable to "
  + "explain more explicitly how the identified profiles translate into intervention plans and "
  + "what variables or patterns trigger specific decisions."
));
children.push(responseLabel());
children.push(body(
  "Please see our response to the Editor's third comment above, where we present a concrete, "
  + "data-grounded framework linking profile characteristics to candidate institutional "
  + "responses, clarify that the underlying covariates (not cluster membership itself) should "
  + "serve as the operational trigger for any decision, and explicitly frame the proposed "
  + "responses as hypotheses for piloting rather than validated recommendations, consistent "
  + "with the cross-sectional nature of the design."
));

// ================= REVIEWER 2 =================
children.push(heading("Response to Reviewer 2", HeadingLevel.HEADING_1));

children.push(heading("Reviewer 2, comment 1 (dependence of cluster structure on UMAP; calibration against null data)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "The cluster structure may depend strongly on UMAP... To address this issue, authors can "
  + "calibrate the complete tuned pipeline against repeated null datasets. If the observed "
  + "solution does not clearly outperform the null benchmarks, the apparent separation may "
  + "reflect the dimensionality-reduction and model-selection procedure rather than meaningful "
  + "cluster structure in the original data."
));
children.push(responseLabel());
children.push(body(
  "We thank the reviewer for this suggestion and implemented the recommended calibration. We "
  + "generated null datasets by independently permuting each of the 32 input variables across "
  + "students, which preserves each variable's marginal distribution while destroying any real "
  + "multivariate association among variables. We ran the complete pipeline (UMAP + K-Means, "
  + "K = 8) five times on the observed data and ten times on independently generated null "
  + "datasets, evaluating Silhouette, Calinski-Harabasz and Davies-Bouldin on a common 50,000-"
  + "observation evaluation protocol. The null datasets equalled or exceeded the observed data "
  + "on all three indices: mean Silhouette 0.457 (null) versus 0.412 (observed; null exceeded "
  + "the observed mean in 80% of replications), mean Calinski-Harabasz 62,591 versus 54,263 "
  + "(90% of null replications higher), and mean Davies-Bouldin 0.719 versus 0.773 (lower is "
  + "better; null also favourable). This result confirms the reviewer's concern directly: these "
  + "internal validity indices do not, on their own, demonstrate that the eight-cluster solution "
  + "captures multivariate structure beyond what the UMAP + K-Means procedure would produce from "
  + "data with the same marginal distributions but no real dependence among variables. "
  + "Accordingly, we have (i) added this calibration as a new subsection and supplementary "
  + "figure, (ii) removed language that used internal Silhouette/Calinski-Harabasz values as "
  + "primary evidence that the eight profiles reflect genuine latent structure, and (iii) "
  + "reframed the paper's validation argument around convergent evidence that does not depend "
  + "on internal cluster-validity indices: the geographic consistency of profile distributions "
  + "with independently known regional inequalities (Section 5.3, not used in cluster "
  + "construction), and the interpretability and policy relevance of the resulting profiles. We "
  + "now explicitly discuss this limitation of unsupervised profiling in the Limitations section. "
  + "As a methodological check, we repeated the calibration with a stricter permutation "
  + "procedure: rather than permuting the already one-hot-encoded design matrix (which can in "
  + "principle generate invalid category combinations for nominal variables), we permuted the "
  + "original categorical labels of each nominal variable prior to encoding, so that every "
  + "simulated student retains a valid, internally consistent combination of categories. "
  + "Results are consistent with, and if anything slightly stronger than, those reported above: "
  + "mean null Silhouette 0.445 (SD = 0.023) versus 0.412 observed (null exceeded or matched the "
  + "observed mean in all 10 replications under this stricter procedure), mean null "
  + "Davies-Bouldin 0.718 versus 0.773 observed (lower is better; null again favourable), and "
  + "mean null Calinski-Harabasz 57,963 versus 54,263 observed (90% of null replications "
  + "higher). We therefore rule out the encoding artefact as an explanation for our finding and "
  + "report both procedures in the supplementary materials."
));

children.push(heading("Reviewer 2, comment 2 (stability evidence)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "A mean ARI of 0.69 (SD = 0.14) indicates meaningful but imperfect agreement... the current "
  + "robustness analysis varies only the K-Means initialization while holding the UMAP embedding "
  + "fixed. To evaluate the stability of the complete pipeline, the authors should repeat both "
  + "stages using different UMAP random seeds and independently drawn 80,000 observation fitting "
  + "samples, and then assess agreement among the resulting cluster assignments."
));
children.push(responseLabel());
children.push(body(
  "Please see our response to the Editor's second comment above, which reports this analysis "
  + "in full (mean ARI = 0.506, SD = 0.085, across 15 repetitions varying both the UMAP seed and "
  + "the fitting sample independently), together with the requested distribution, a consensus/"
  + "co-clustering matrix, and cluster-specific stability estimates."
));

children.push(heading("Reviewer 2, comment 3 (comparison among clustering algorithms)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Comparison among clustering algorithms. The algorithms were evaluated on different search "
  + "spaces, since DBSCAN was restricted to two dimensions while HDBSCAN's min_cluster_size "
  + "values are very small relative to 200,000 observations. Relevant GMM decisions are not "
  + "reported, including covariance specification and initialization. The authors could "
  + "evaluate comparable dimensionalities and broader hyperparameter ranges before concluding "
  + "that density-based methods are unsuitable or that the data lack variable-density cluster "
  + "structure."
));
children.push(responseLabel());
children.push(body(
  "We thank the reviewer for this observation and addressed all three points by reproducing "
  + "the exact Phase 1 methodology (same 200,000-observation subsample, same seed) with an "
  + "expanded search space. First, we reran DBSCAN across UMAP dimensionality 2-5 (previously "
  + "only 2D was tested), using the same eps estimation and min_samples range as the original "
  + "2D sweep. Noise remained very high (30-99% of points labelled as noise) in every "
  + "dimensionality tested, confirming that this limitation was not an artefact of restricting "
  + "DBSCAN to two dimensions. Second, we expanded the HDBSCAN min_cluster_size range from "
  + "{5, 10, 15} — under 0.01% of the 200,000-observation sample — to include values more "
  + "appropriate to this sample size (50, 100, 250, 500, 1000). With min_cluster_size = 250-500 "
  + "(approximately 0.1-0.25% of the sample), HDBSCAN converges independently to exactly eight "
  + "clusters, with very low noise (0.2-1.5%), consistently across all four dimensionalities "
  + "tested (2D, 3D, 4D, 5D). We view this as valuable convergent evidence for the eight-cluster "
  + "solution, independent of K-Means, and we have added it to our response to the Editor's "
  + "first comment and to Section 4.2 as additional support for K = 8. Third, we report the "
  + "previously undocumented GMM specification (covariance_type = 'full', n_init = 1) and "
  + "tested its sensitivity by running all four available covariance types with n_init of 1 "
  + "and 10 at both K = 2 and K = 8. Results are robust to this choice (Silhouette ranges "
  + "0.40-0.46 at K = 8 across all eight configurations); BIC improves modestly with n_init = 10 "
  + "regardless of covariance type, which we now adopt as the reported configuration, but no "
  + "conclusion in the manuscript changes as a result. We have added a new supplementary figure "
  + "and three supplementary tables (DBSCAN, HDBSCAN, GMM sensitivity) reporting these results "
  + "in full."
));

children.push(heading("Reviewer 2, comment 4 (missing data)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Missing data. Maternal and paternal education show approximately 22-23% missing values "
  + "each, and both are replaced with a single median value. This can create artificial "
  + "concentrations, attenuate variance, and alter UMAP neighborhoods. The case check for "
  + "Clusters 1 and 5 is useful but does not evaluate effects on the full partition. Please "
  + "repeat the analysis using multiple imputation, missingness indicators, or another "
  + "principled strategy, and compare the resulting cluster assignments."
));
children.push(responseLabel());
children.push(body(
  "We implemented both alternatives the reviewer suggested and compared the resulting "
  + "partitions against the original (single-median) assignments. Missingness is 23.4% for "
  + "paternal education and 21.6% for maternal education. Using three independent seeds, each "
  + "holding the fitting sample, evaluation set and UMAP/K-Means random state fixed so that "
  + "only the missing-data treatment varies, we compared: (i) adding two binary missingness "
  + "indicators alongside the median-imputed values, and (ii) replacing the median imputation "
  + "with multiple imputation (IterativeImputer with a Bayesian ridge estimator, using the "
  + "other ordinal and achievement variables as predictors, redrawn independently for each "
  + "seed). Relative to the baseline (median-only) assignments on the same seed, the "
  + "missingness-indicator variant yields a mean ARI of 0.464 (SD = 0.005), close to the "
  + "0.506 pipeline-to-pipeline variability we report in response to Editor comment 2 purely "
  + "from re-running the pipeline with a different seed — indicating that adding indicators "
  + "changes the partition by roughly as much as ordinary pipeline stochasticity. The "
  + "multiple-imputation variant, however, yields a substantially lower mean ARI of 0.274 "
  + "(SD = 0.033) relative to the same baseline, clearly below the 0.506 noise floor. This is "
  + "an honest and, we believe, important finding: the choice of missing-data strategy for "
  + "parental education materially changes the resulting eight-cluster partition, beyond what "
  + "ordinary UMAP/K-Means randomness would explain, confirming the reviewer's concern. We "
  + "have added this comparison as a new supplementary figure and table, and revised the "
  + "Limitations section to state explicitly that the reported profiles are sensitive to the "
  + "missing-data treatment for parental education; we now present the multiple-imputation "
  + "result as a sensitivity analysis alongside the original median-based partition, rather "
  + "than treating the latter as unconditionally robust."
));

children.push(heading("Reviewer 2, comment 5 (mean differences reported as validation)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Reported mean differences and clustering construction. Academic achievement and the "
  + "socioeconomic variables are used to construct the clusters, so the differences in their "
  + "cluster means and the associated tests are not independent findings. With n = 452,020, "
  + "even small differences will be statistically significant. These results should be "
  + "presented descriptively with effect sizes, not as validation. Independent validation "
  + "requires variables that were excluded from the construction of the clustering."
));
children.push(responseLabel());
children.push(body(
  "We agree with this point. The tables and figures reporting mean differences across "
  + "clusters on academic-performance and socioeconomic variables use exactly the variables "
  + "that entered the UMAP embedding used to construct the clusters, so these comparisons "
  + "cannot serve as independent validation of the partition; with n = 452,020, essentially "
  + "any nonzero difference is also statistically significant, which makes p-values "
  + "uninformative in this setting. We have revised these sections to present the comparisons "
  + "purely descriptively: we removed language describing them as 'confirming' or 'validating' "
  + "the clusters, replaced the emphasis on statistical significance with standardized effect "
  + "sizes (Cohen's d for pairwise cluster comparisons, eta-squared for the overall by-cluster "
  + "comparison), and added an explicit statement that independent validation in this study "
  + "relies instead on variables not used in cluster construction, namely the geographic "
  + "distribution of profiles (Section 5.3)."
));

children.push(heading("Reviewer 2, comment 6 (clustering contribution beyond a simple tabulation)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Clustering contribution beyond simpler analyses. The headline finding that Clusters 0 and "
  + "6 have similar means but different constraints amounts to a cross-tabulation of scores by "
  + "hours worked and internet access. For a general audience, the paper should show that "
  + "profile membership carries information beyond a linear combination of the inputs, for "
  + "example by predicting an outcome (score, or ideally graduation) conditional on the "
  + "covariates."
));
children.push(responseLabel());
children.push(body(
  "We implemented the test the reviewer proposes. Our dataset does not include a graduation "
  + "indicator, so we used the global Saber Pro score (PUNT_GLOBAL) as the outcome. We took "
  + "care to avoid a circularity: PUNT_GLOBAL is essentially a function of the five achievement "
  + "modules that were themselves inputs to the published clustering, so adding published "
  + "cluster membership to a model predicting PUNT_GLOBAL would show an inflated, invalid "
  + "improvement driven by information leakage rather than by the profiles capturing genuine "
  + "structure. To test the reviewer's question fairly, we built an alternative clustering "
  + "(K = 8, same UMAP + K-Means pipeline) using only the socioeconomic and access covariates, "
  + "excluding the five achievement modules entirely. On a common 50,000-observation evaluation "
  + "set with five-fold cross-validation, a Ridge regression of PUNT_GLOBAL on the "
  + "socioeconomic/access covariates alone achieves R² = 0.174; adding the socioeconomic-only "
  + "cluster membership as an additional predictor raises this to R² = 0.174 "
  + "(ΔR² ≈ 0.001, not meaningful). A more flexible, nonlinear baseline "
  + "(HistGradientBoostingRegressor) on the same covariates alone achieves R² = 0.235, "
  + "essentially unchanged (R² = 0.235) when cluster membership is added. For comparison, "
  + "adding the published (score-informed) cluster membership to the linear baseline raises "
  + "R² to 0.243 (ΔR² ≈ 0.069), but we report this only to illustrate the scale of the "
  + "leakage effect, not as valid evidence, since the published clusters were partly "
  + "constructed from the outcome being predicted. In the fair, non-circular test, cluster "
  + "membership does not carry predictive information beyond a linear or nonlinear combination "
  + "of the same covariates. This is an honest finding that confirms the reviewer's concern: "
  + "we have revised the manuscript to reframe the contribution of the clustering as "
  + "descriptive and exploratory — a typology intended to support interpretable, policy-facing "
  + "communication of how structural constraints co-occur — rather than as a source of "
  + "additional predictive power over the same covariates, and we removed language implying "
  + "the latter. We added this analysis and the corresponding figure as a new supplementary "
  + "section."
));

children.push(heading("Reviewer 2, comment 7 (Figure 3 misread as P(department | cluster))", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Geographic analysis and external validation. Each row of Figure 3 sums to approximately "
  + "100%, so its cells report the percentage of examinees within each department assigned to "
  + "each cluster, P(cluster | department). Several statements appear instead to interpret them "
  + "as P(department | cluster). In particular, the 16% reported for Bogotá and the 14% for "
  + "Valle del Cauca correspond to Cluster 4 in the figure, while the values shown for Cluster 7 "
  + "are 2% and 1%, respectively. For example, the claim that Cluster 7 is concentrated in these "
  + "urban centres is therefore not supported by the figure as presented."
));
children.push(responseLabel());
children.push(body(
  "We thank the reviewer for identifying this precisely, and we confirm the error: Figure 3 is "
  + "row-normalized by department and therefore reports P(cluster | department), not "
  + "P(department | cluster). The claim that Cluster 7 is concentrated in Bogotá and Valle del "
  + "Cauca was incorrect under the correct reading (2% and 1% for Cluster 7 versus 16% and 14% "
  + "for Cluster 4, as the reviewer notes) and has been removed. We corrected the text "
  + "accompanying Figure 3 and reviewed every other passage in the manuscript that references "
  + "this figure, correcting the same P(cluster | department) versus P(department | cluster) "
  + "confusion wherever it occurred and removing any claim that is not supported once the "
  + "figure is read correctly. We also added one clarifying sentence directly under the figure "
  + "stating explicitly what each row and cell represents, to prevent this misreading going "
  + "forward."
));

children.push(heading("Reviewer 2, comment 8 (temporal analysis)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Temporal analysis. The analysis pools administrations from 2021-2 through 2023-1, but does "
  + "not show that score scales, variable distributions, and profile prevalence are comparable "
  + "across periods. Please document comparability and report cluster composition by "
  + "administration. Fitting the pipeline separately for each period would provide an important "
  + "temporal replication test and reveal whether the profiles are stable rather than "
  + "cohort-specific."
));
children.push(responseLabel());
children.push(body(
  "We addressed all three requests. First, we compared score scales and distributions across "
  + "the four administrations (2021-2, n = 150,444; 2022-2, n = 75,694; 2022-5, n = 103,777; "
  + "2023-1, n = 122,105): means and standard deviations of the global score and the five "
  + "modules are similar across periods, with differences of a few points against a standard "
  + "deviation of approximately 23-27, and no systematic drift. Second, we report the "
  + "prevalence of each published cluster by administration; prevalence is reasonably stable "
  + "(for example, Cluster 0 ranges 15.1-17.2% and the smallest cluster ranges 0.5-1.4% across "
  + "all four periods), which we now present as a new supplementary table. Third, and most "
  + "importantly, we implemented the requested temporal replication test: we refit the complete "
  + "UMAP + K-Means pipeline (K = 8) independently within each of the four administrations, "
  + "without pooling data across periods, and matched the resulting clusters against the "
  + "2021-2 solution by centroid similarity in the original 32-variable feature space (Hungarian "
  + "algorithm). The mean correlation between matched centroids is 0.956 (2022-2), 0.955 "
  + "(2022-5), and 0.917 (2023-1); the large majority of individual clusters match at a "
  + "correlation of 0.95 or higher, with one or two smaller, more niche clusters matching less "
  + "strongly (0.76-0.87). We interpret this as evidence that the same profiles re-emerge "
  + "independently in each administration and are not an artefact specific to a single cohort, "
  + "with the caveat that the smallest profiles replicate somewhat less precisely. We have "
  + "added this analysis, the comparability statistics, and the by-period composition table as "
  + "a new supplementary section and figure."
));

children.push(heading("Reviewer 2, comment 9 (sample coverage and representativeness)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Sample. The similarity between a 200,000-record subsample and the dataset from which it was "
  + "drawn does not establish national representativeness. The authors can describe coverage, "
  + "exclusions, and possible selection into Saber Pro participation. They should also clarify "
  + "whether the 452,020 observations represent unique students or examination records that may "
  + "include repeat presenters."
));
children.push(responseLabel());
children.push(body(
  "We have added a paragraph to the Data section describing sample coverage and clarifying the "
  + "unit of observation. The 452,020 records correspond to Saber Pro examination "
  + "administrations from 2021-2 through 2023-1 (four administrations); Saber Pro is required "
  + "of students completing an accredited undergraduate program in Colombia during that window, "
  + "so the sample corresponds to a census of graduating-cohort examinees for this period rather "
  + "than a survey drawn from a broader population. Using the anonymized student identifier "
  + "(ESTU_CONSECUTIVO), we confirm that repeated presenters are rare: 1,051 of 452,020 records "
  + "(0.23%) share an identifier with another record in the dataset, corresponding to 450,969 "
  + "unique identifiers. We now report this figure explicitly, clarify that our unit of analysis "
  + "is the examination record (consistent with prior Saber Pro literature), and flag the small "
  + "number of repeat presenters as a minor limitation. We also added a sentence acknowledging "
  + "that our sample does not include students who did not complete their program, or who "
  + "completed it without sitting Saber Pro, which is itself a form of selection: our findings "
  + "describe inequality among students who completed a program and took the exam, not among "
  + "all students who initially enrolled."
));

children.push(heading("Reviewer 2, comment 10 (institutional/programme-level concentration)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Institutional and programme-level concentration as a plausible alternative explanation. "
  + "Students are nested within programmes and institutions, but these structures are neither "
  + "examined nor mentioned as a limitation. Geographic aggregation does not demonstrate that "
  + "the profiles are independent of a small number of institutions or fields of study. At "
  + "minimum, the authors could report institutional and programme composition by cluster, and "
  + "conduct a leave-institution-out, leave-programme-out, or comparable sensitivity analysis. "
  + "This is important for interpreting the profiles as student-level rather than "
  + "institutional configurations."
));
children.push(responseLabel());
children.push(body(
  "We carried out both analyses the reviewer requests. First, we report institutional and "
  + "programme composition within each cluster. No cluster is dominated by a single "
  + "institution or programme: each of the eight clusters contains between 138 and 246 of the "
  + "265 institutions in the sample, and between 160 and 467 of the 953 programmes, with the "
  + "largest institution representing between 3.4% and 11.6% of any given cluster "
  + "(concentration index, or HHI, of 0.011-0.037, close to the sample-wide reference value of "
  + "0.014). We view this as evidence that the profiles reflect student-level configurations "
  + "rather than a small number of institutions or fields of study. Second, we implemented a "
  + "leave-institution-out / leave-programme-out sensitivity analysis: we refit the complete "
  + "pipeline ten times, excluding each of the five largest institutions and the five largest "
  + "programmes one at a time (up to 29,499 students, or 6.5% of the sample, excluded in the "
  + "largest case), and compared the resulting partition on the remaining students against the "
  + "published partition using the Adjusted Rand Index. The mean ARI across all ten exclusions "
  + "is 0.449 (SD = 0.047, range 0.347-0.527), which falls within the normal pipeline-to-"
  + "pipeline variability we report in response to Editor comment 2 (0.506, SD = 0.085) for all "
  + "but one exclusion: removing the single largest institution (29,499 students) produces a "
  + "somewhat lower ARI of 0.347, outside that normal range. We report this as a minor "
  + "limitation: while the profiles are broadly robust to excluding any single large "
  + "institution or programme, the very largest institution in the sample may have a "
  + "modestly disproportionate influence on the overall structure. We have added the "
  + "composition and sensitivity results as new supplementary tables and a figure, and revised "
  + "the Limitations section to note this exception explicitly."
));

children.push(heading("Reviewer 2, comment 11 (Cluster 7's disconnected UMAP islands)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Cluster 7 deserves closer examination. Cluster 7 contains only 1.5% of observations and "
  + "forms two disconnected islands in the UMAP representation. It would be very valuable for "
  + "the authors to assess whether this reflects reproducible subpopulations, extreme values, "
  + "or scoring or preprocessing errors. The two islands could be characterized separately."
));
children.push(responseLabel());
children.push(body(
  "We located and characterized this cluster in our reproduction (536 of 50,000 evaluation "
  + "observations, 1.07%, closely matching the 1.5% reported in the manuscript). It is not an "
  + "artefact of extreme or invalid values: its members are students with very high scores "
  + "(mean global score 197-212, against a sample-wide mean of 146.8), high internet access "
  + "(91-99%), concentrated in a comparatively small number of institutions (53-113 distinct "
  + "institutions, versus 200 or more for the other clusters) — consistent with a genuine "
  + "high-achieving subgroup, plausibly linked to more selective institutions, rather than a "
  + "data error. Consistent with the manuscript, this cluster does form its own well-separated "
  + "island relative to the rest of the embedding, with a clear gap from the main body of "
  + "points, which supports treating it as a genuinely distinctive subgroup rather than noise. "
  + "However, on close examination of the 536 points themselves (DBSCAN at several "
  + "neighbourhood radii found no natural split), we did not reproduce two disconnected "
  + "islands within this cluster: the points form a single, continuous, elongated cloud rather "
  + "than two separate components. We were nonetheless able to identify two subgroups along "
  + "this continuum (a smaller, higher-scoring group of 215 students concentrated in 53 "
  + "institutions, and a larger group of 321 students spread across 113 institutions), which we "
  + "report and characterize separately as requested, while noting that they are not cleanly "
  + "disconnected in our reproduction. We interpret the discrepancy with the originally "
  + "reported two-island appearance as most likely specific to the exact geometry of that "
  + "particular UMAP run, consistent with the pipeline instability we document in response to "
  + "Editor comment 2 (mean ARI ≈ 0.51 across reruns) — small clusters are especially "
  + "susceptible to this kind of run-specific visual variation even when the broader "
  + "eight-cluster structure is preserved. Reassuringly, our existing stability analysis shows "
  + "this specific cluster has a moderate-to-good reproducibility (69.3% mean agreement with "
  + "the reference labelling across 15 reruns), roughly in the middle of the eight clusters, "
  + "supporting that it is a real, fairly reproducible group rather than noise, even though its "
  + "precise visual shape (one compact island versus two) is not robust across runs. We have "
  + "revised the manuscript to characterize this cluster's two subgroups and to soften the "
  + "'two disconnected islands' description to note that it reflects one particular run rather "
  + "than a robust feature, and added the corresponding figure."
));

children.push(heading("Reviewer 2, comment 12 (invalid Saber Pro vs. PISA comparison; hypotheses presented as findings)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "Comparisons and overinterpretation. The comparison between the Saber Pro gap and previously "
  + "reported PISA gaps is not valid, because the two assessments use different scales and "
  + "standard deviations; standardized differences should be reported instead. In addition, the "
  + "statements in Section 6 regarding structural constraints, labour-market mechanisms, digital "
  + "exclusion, and maternal or paternal cultural channels should be explicitly identified as "
  + "hypotheses rather than as findings of the cross-sectional analysis."
));
children.push(responseLabel());
children.push(body(
  "We agree on both points. We removed the direct numerical comparison between the Saber Pro "
  + "score gap and previously reported PISA gaps, since the two instruments differ in scale and "
  + "standard deviation and are not directly comparable; we replaced it with a standardized "
  + "(SD-unit) difference computed entirely within our own data, with no reference to PISA. We "
  + "also revised Section 6 throughout so that statements about structural constraints, "
  + "labour-market mechanisms, digital exclusion, and parental cultural channels are explicitly "
  + "labelled as interpretive hypotheses consistent with the observed descriptive patterns, "
  + "rather than as conclusions established by the analysis, which is cross-sectional and cannot "
  + "identify the causal or mechanistic pathways these statements describe."
));

children.push(heading("Reviewer 2, comment 13 (speculative intervention proposals)", HeadingLevel.HEADING_2));
children.push(commentPara(
  "In the conclusions, the three proposed intervention responses are speculative; they could be "
  + "presented as a policy-oriented framework, but their effectiveness and appropriateness "
  + "require causal evaluation."
));
children.push(responseLabel());
children.push(body(
  "We agree and have revised the Conclusions accordingly. The three intervention responses are "
  + "now explicitly framed as a speculative, policy-oriented framework motivated by the "
  + "descriptive cluster profiles, rather than as validated recommendations. We added a sentence "
  + "stating directly that their effectiveness and appropriateness have not been evaluated here "
  + "and would require dedicated causal evaluation (for example, pilot studies or "
  + "quasi-experimental designs) before implementation."
));

const doc = new Document({
  sections: [{
    properties: { page: { size: PAGE.size, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 22 } },
    },
  },
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/Respuesta_a_revisores_borrador.docx", buf);
  console.log("saved");
});
